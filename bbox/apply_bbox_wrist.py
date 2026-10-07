#!/usr/bin/env python3
"""손목 카메라 bbox 슬롯을 하나 더 붙인다.

프롬프트
  State: ...; Mistake: false; Pen:<4토큰> Wrist:<4토큰>; Subtask: ...;\\nAction:
         └ cam_high ┘        └ cam_right_wrist ┘

왜 두 개인가 (실측 327,000 프레임)
  cam_high 만      10.9%
  손목 만          14.8%   <- high 가 펜을 놓치는 구간을 메운다
  둘 다            74.2%
  **둘 다 없음      0.0%**
  구간별로는 왼팔 운반에서 high 70% / 손목 100%, 꽂기에서 high 79% / 손목 56% 다.
  두 뷰가 서로를 메운다.

슬롯 배치
  head("... Pen:") | pen 4 | " Wrist:" 2 | wrist 4 | "; Subtask:" 4 | subtask 14 | tail
  사이 텍스트의 토큰 수가 전부 상수라 CE·디코드가 오프셋으로 찾을 수 있다.

켜기: PI05_BBOX_WRIST=1  (PI05_BBOX=1 이 전제다. 학습·추론 양쪽에 같은 값)
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import migrate_v2

MARK = "RSC_WBOX"


def rep(root, rel, pairs, scope=None):
    p = root / rel
    s = p.read_text()
    if MARK in s:
        print(f"  {rel}: 이미 적용됨"); return
    lo, hi = 0, len(s)
    if scope:
        lo = s.index(scope)
        nxt = s.find("\nclass ", lo + 1)
        hi = nxt if nxt != -1 else len(s)
    body = s[lo:hi]
    for a, _ in pairs:
        n = body.count(a)
        if n != 1:
            raise SystemExit(f"★ {rel}: 앵커가 {n} 번 (1 이어야 함)\n{a.splitlines()[0][:90]}")
    for a, b in pairs:
        body = body.replace(a, b, 1)
    p.write_text(s[:lo] + body + s[hi:])
    print(f"  {rel}: ✓ 적용")


# ── ① tokenizer ───────────────────────────────────────────────────────────
TOK = "src/openpi/models/tokenizer.py"
TOK_P = [
 ('    BBOX_MID = "; Subtask:"',
  '    BBOX_MID = "; Subtask:"\n'
  f'    # {MARK}: 손목 슬롯. " Wrist:" 는 2토큰이다 (CE·디코드 오프셋 상수).\n'
  '    WBOX_MID = " Wrist:"'),

 ('    def tokenize_with_subtask(self, prompt: str, state: np.ndarray, subtask: str,\n'
  '                              mistake: bool | None = None,\n'
  '                              bbox: "np.ndarray | None" = None):',
  '    def tokenize_with_subtask(self, prompt: str, state: np.ndarray, subtask: str,\n'
  '                              mistake: bool | None = None,\n'
  '                              bbox: "np.ndarray | None" = None,\n'
  '                              wbox: "np.ndarray | None" = None):'),

 # wmid/wbox_body 는 **분기 밖**에서 먼저 비워 둬야 한다. bbox=None(대조군) 이면
 # 아래 블록이 안 돌아 UnboundLocalError 가 난다.
 ('        pad_id0 = self._tokenizer.pad_id() if self._tokenizer.pad_id() >= 0 else 0',
  '        pad_id0 = self._tokenizer.pad_id() if self._tokenizer.pad_id() >= 0 else 0\n'
  f'        wbox_body, wmid = [], []        # {MARK}: 분기 밖 초기화'),

 ('            mid = self._tokenizer.encode(self.BBOX_MID)',
  f'            # {MARK}: 손목 슬롯이 있으면 pen 과 "; Subtask:" 사이에 끼운다.\n'
  '            if wbox is not None:\n'
  '                w = np.asarray(wbox).reshape(-1).astype(int)\n'
  '                if w.size >= 5 and w[4] > 0:\n'
  '                    base = self._tokenizer.piece_to_id("<loc0000>")\n'
  '                    wbox_body = [base + int(min(max(v, 0), 1023)) for v in w[:4]]\n'
  '                else:\n'
  '                    wbox_body = [pad_id0] * self.BBOX_SLOT      # 추론: 빈 슬롯\n'
  '                wmid = self._tokenizer.encode(self.WBOX_MID)\n'
  '            else:\n'
  '                wbox_body, wmid = [], []      # 손목 끔\n'
  '            mid = self._tokenizer.encode(self.BBOX_MID)'),

 ('        tokens = head + bbox_body + mid + body + tail\n'
  '        lo_bbox = len(head)\n'
  '        lo = lo_bbox + len(bbox_body) + len(mid)',
  f'        # {MARK}: head | pen 4 | " Wrist:" | wrist 4 | "; Subtask:" | subtask 14 | tail\n'
  '        tokens = head + bbox_body + wmid + wbox_body + mid + body + tail\n'
  '        lo_bbox = len(head)\n'
  '        lo_wbox = lo_bbox + len(bbox_body) + len(wmid)\n'
  '        lo = lo_wbox + len(wbox_body) + len(mid)'),

 ('        loss = [(lo <= i < lo + self.SUBTASK_SLOT)\n'
  '                or (lo_bbox <= i < lo_bbox + len(bbox_body))\n'
  '                for i in range(self._max_len)]',
  f'        # {MARK}: 세 슬롯을 모두 덮는다 (구조용 — 라벨 유무와 무관).\n'
  '        loss = [(lo <= i < lo + self.SUBTASK_SLOT)\n'
  '                or (lo_bbox <= i < lo_bbox + len(bbox_body))\n'
  '                or (lo_wbox <= i < lo_wbox + len(wbox_body))\n'
  '                for i in range(self._max_len)]'),
]

# ── ② transforms ──────────────────────────────────────────────────────────
TR = "src/openpi/transforms.py"
TR_P = [
 ('    labels_dir: str = ""\n\n    def __call__(self, data: DataDict) -> DataDict:\n'
  '        if not self.labels_dir or "episode_index" not in data or "frame_index" not in data:',
  '    labels_dir: str = ""\n'
  f'    # {MARK}: 손목 카메라 라벨. 비면 손목 슬롯을 안 만든다.\n'
  '    wrist_labels_dir: str = ""\n\n    def __call__(self, data: DataDict) -> DataDict:\n'
  '        if not self.labels_dir or "episode_index" not in data or "frame_index" not in data:'),

 ('        return {**data, "bbox": np.asarray(t[fr], np.int32)}',
  '        out = {**data, "bbox": np.asarray(t[fr], np.int32)}\n'
  f'        # {MARK}\n'
  '        if self.wrist_labels_dir:\n'
  '            q = pathlib.Path(self.wrist_labels_dir) / f"ep{ep:04d}.npy"\n'
  '            if q.exists():\n'
  '                u = np.load(q, mmap_mode="r")\n'
  '                if fr < len(u):\n'
  '                    out["wbox"] = np.asarray(u[fr], np.int32)\n'
  '        return out'),

 ('    bbox_slot: bool = False',
  '    bbox_slot: bool = False\n'
  f'    # {MARK}: True 면 손목 슬롯도 만든다. 학습·추론이 같아야 한다.\n'
  '    wbox_slot: bool = False'),

 ('        if not self.bbox_slot:\n            bbox = None\n'
  '        elif bbox is None:\n            bbox = np.zeros(5, np.int32)',
  '        if not self.bbox_slot:\n            bbox = None\n'
  '        elif bbox is None:\n            bbox = np.zeros(5, np.int32)\n'
  f'        # {MARK}: 손목도 같은 규칙 — 끄면 라벨이 와도 버린다.\n'
  '        wbox = data.pop("wbox", None)\n'
  '        if not self.wbox_slot:\n'
  '            wbox = None\n'
  '        elif wbox is None:\n'
  '            wbox = np.zeros(5, np.int32)'),

 ('                prompt, state, subtask, mistake=mistake, bbox=bbox)',
  '                prompt, state, subtask, mistake=mistake, bbox=bbox, wbox=wbox)'),
]

# ── ③ libero_policy ───────────────────────────────────────────────────────
POL = "src/openpi/policies/libero_policy.py"
POL_P = [
 ('        if "bbox" in data:\n            inputs["bbox"] = data["bbox"]',
  '        if "bbox" in data:\n            inputs["bbox"] = data["bbox"]\n'
  f'        # {MARK}\n'
  '        if "wbox" in data:\n            inputs["wbox"] = data["wbox"]'),
]

# ── ④ pi0.py — CE 오프셋 / 디코드 ─────────────────────────────────────────
PI0 = "src/openpi/models/pi0.py"
PI0_P = [
 ('_RSC_BBOX_MID = 4',
  '_RSC_BBOX_MID = 4\n'
  f'# {MARK}: 손목 슬롯. " Wrist:" 는 2토큰.\n'
  '_RSC_WBOX_ON = _os.environ.get("PI05_BBOX_WRIST", "0") == "1"\n'
  '_RSC_WBOX_SLOT = 4\n'
  '_RSC_WBOX_MID = 2'),

 ('        if _RSC_BBOX_ON:\n'
  '            off_b = jnp.arange(_RSC_BBOX_SLOT)\n'
  '            off_s = _RSC_BBOX_SLOT + _RSC_BBOX_MID + jnp.arange(_RSC_SUBTASK_SLOT)\n'
  '            offs = jnp.concatenate([off_b, off_s])\n'
  '            n_bbox = _RSC_BBOX_SLOT',
  f'        # {MARK}: 슬롯이 셋이면 pen | " Wrist:" | wrist | "; Subtask:" | subtask 다.\n'
  '        if _RSC_BBOX_ON:\n'
  '            off_b = jnp.arange(_RSC_BBOX_SLOT)\n'
  '            if _RSC_WBOX_ON:\n'
  '                w0 = _RSC_BBOX_SLOT + _RSC_WBOX_MID\n'
  '                off_w = w0 + jnp.arange(_RSC_WBOX_SLOT)\n'
  '                s0 = w0 + _RSC_WBOX_SLOT + _RSC_BBOX_MID\n'
  '                offs = jnp.concatenate([off_b, off_w, s0 + jnp.arange(_RSC_SUBTASK_SLOT)])\n'
  '                n_bbox = _RSC_BBOX_SLOT + _RSC_WBOX_SLOT\n'
  '            else:\n'
  '                off_s = _RSC_BBOX_SLOT + _RSC_BBOX_MID + jnp.arange(_RSC_SUBTASK_SLOT)\n'
  '                offs = jnp.concatenate([off_b, off_s])\n'
  '                n_bbox = _RSC_BBOX_SLOT'),

 ('        if _RSC_BBOX_ON:\n'
  '            done0 = jnp.zeros(tok.shape[0], dtype=bool)\n'
  '            for k in range(_RSC_BBOX_SLOT):\n'
  '                tok, done0 = one(tok, done0, k, always_open=True)\n'
  '            sub0 = _RSC_BBOX_SLOT + _RSC_BBOX_MID\n'
  '        else:\n'
  '            sub0 = 0',
  f'        # {MARK}: 슬롯 순서대로 채운다. 길이가 고정이라 EOS 를 보지 않는다.\n'
  '        if _RSC_BBOX_ON:\n'
  '            done0 = jnp.zeros(tok.shape[0], dtype=bool)\n'
  '            for k in range(_RSC_BBOX_SLOT):\n'
  '                tok, done0 = one(tok, done0, k, always_open=True)\n'
  '            if _RSC_WBOX_ON:\n'
  '                w0 = _RSC_BBOX_SLOT + _RSC_WBOX_MID\n'
  '                for k in range(_RSC_WBOX_SLOT):\n'
  '                    tok, done0 = one(tok, done0, w0 + k, always_open=True)\n'
  '                sub0 = w0 + _RSC_WBOX_SLOT + _RSC_BBOX_MID\n'
  '            else:\n'
  '                sub0 = _RSC_BBOX_SLOT + _RSC_BBOX_MID\n'
  '        else:\n'
  '            sub0 = 0'),
]

# ── ⑤ config ──────────────────────────────────────────────────────────────
CFG = "src/openpi/training/config.py"
CFG_P = [
 ('    bbox_labels_dir: str = ""',
  '    bbox_labels_dir: str = ""\n'
  f'    # {MARK}: 손목 카메라 라벨 (순기구학 투영). 비면 손목 슬롯을 안 만든다.\n'
  '    wrist_bbox_labels_dir: str = ""'),
 ('                inputs=[_transforms.InjectBBox(labels_dir=self.bbox_labels_dir)],',
  '                inputs=[_transforms.InjectBBox(\n'
  '                    labels_dir=self.bbox_labels_dir,\n'
  '                    wrist_labels_dir=self.wrist_bbox_labels_dir)],'),
 ('                            bbox_slot=_os.environ.get("PI05_BBOX", "0") == "1",',
  '                            bbox_slot=_os.environ.get("PI05_BBOX", "0") == "1",\n'
  f'                            # {MARK}: 학습·추론에 같은 값이 들어가야 한다.\n'
  '                            wbox_slot=_os.environ.get("PI05_BBOX_WRIST", "0") == "1",'),
]


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    # ★ 마커 검사보다 **먼저**. 옛 커밋이 심어 둔 코드는 마커가 이미 있어
    #   아래가 전부 "이미 적용됨" 으로 끝난다 — 그 전에 옛 코드를 새 코드로 올린다.
    migrate_v2.run(root, ["src/openpi/models/tokenizer.py"])
    rep(root, TOK, TOK_P)
    rep(root, TR,  TR_P)
    rep(root, POL, POL_P, scope="class EmbodiChainInputs")
    rep(root, PI0, PI0_P)
    rep(root, CFG, CFG_P)
    print("WBOX-PATCH-DONE")


if __name__ == "__main__":
    main()
