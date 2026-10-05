#!/usr/bin/env python3
"""펜 위치 grounding — PaliGemma 검출 형식 bbox 슬롯을 프롬프트에 끼운다.

프롬프트
  State: 101 134 ...; Mistake: false; Pen:<loc0296><loc0916><loc0492><loc1023>
  ; Subtask: lower the right gripper onto the pen;
  Action:

왜 이 형식인가
  PaliGemma 는 <loc_ymin><loc_xmin><loc_ymax><loc_xmax> **그 순서** 로 검출을
  사전학습했다. 토큰은 사전에 이미 있고(id 256000~257023) 각 1토큰이다.
  어휘뿐 아니라 의미와 순서까지 재사용하려면 그대로 따라야 한다.

  Subtask **앞** 에 온다 — π0.5 도 bbox 를 subtask 전에 예측한다. 슬롯이 causal
  이라 "펜이 여기 있다" 를 먼저 말하고 그 조건 위에서 subtask 를, 다시 그 위에서
  action 을 낸다. 토큰이 시퀀스 안에 있으므로 action expert 가 KV 로 읽는다.

손실
  subtask 슬롯과 같은 CE 다 (token_loss_mask 가 두 슬롯을 모두 덮는다).
  v1 은 두 슬롯에 **같은 가중치**(PI05_SUBTASK_W)를 쓴다. 따로 주고 싶으면
  per-token 가중치 배열이 필요한데 Observation 에 필드를 더해야 해서 미뤘다.
  박스가 안 보이는 프레임(15%)은 슬롯이 패딩이고 loss 에서 빠진다.

켜기: PI05_BBOX=1  (학습·추론 양쪽에 같은 값)
"""
import pathlib, sys

MARK = "RSC_BBOX"

def _norm(l):
    """줄 끝의 `,` `)` `:` 와 공백을 떼고 비교한다.

    패치끼리 **같은 줄**(함수 시그니처 등)에 인자를 덧붙이면, 먼저 적용된 패치가
    넣은 줄의 끝이 `= None):` 에서 `= None,` 로 바뀐다. 통째로 비교하면
    '아직 적용 안 됨' 으로 오판하고 사라진 앵커를 찾다 죽는다. (2026-10-05)
    """
    return l.strip().rstrip(",):").strip()


def rep(text, old, new, *, scope=None, path="", mark=MARK):
    oldn = {_norm(l) for l in old.splitlines()}
    # 주석은 판정에서 뺀다 — 패치마다 MARK 문자열이 달라 같은 코드를 '미적용' 으로 본다.
    added = [_norm(l) for l in new.splitlines()
             if l.strip() and not l.strip().startswith("#") and _norm(l) not in oldn]
    hay = text
    if scope and scope in text:
        i = text.index(scope); j = text.find("\nclass ", i + 1)
        hay = text[i: len(text) if j < 0 else j]
    hayn = [_norm(l) for l in hay.splitlines()]
    # 정확히 같은 줄이 아니라 **부분 문자열** 로 본다. 패치끼리 같은 호출에
    # 인자를 덧붙이면 `f(a, b)` 가 `f(a, b, c)` 가 되어 완전 일치가 깨진다.
    def _has(ln):
        return any(ln and ln in h for h in hayn)
    if added and all(_has(ln) for ln in added):
        return text, False
    body, lo = text, 0
    if scope:
        i = text.index(scope); j = text.find("\nclass ", i + 1)
        body, lo = text[i: len(text) if j < 0 else j], i
    n = body.count(old)
    if n != 1:
        raise SystemExit(f"★ {path}: 앵커가 {n} 번 (1 이어야 함)\n{old.splitlines()[0][:100]}")
    k = lo + body.index(old)
    return text[:k] + new + text[k + len(old):], True


# ── 1. tokenizer.py ────────────────────────────────────────────────────────
TOK = "src/openpi/models/tokenizer.py"
TOK_P = [
 ('    SUBTASK_SLOT = 14',
  '    SUBTASK_SLOT = 14\n'
  f'    # {MARK}: PaliGemma 검출 형식 — ymin, xmin, ymax, xmax 네 토큰.\n'
  '    BBOX_SLOT = 4\n'
  '    # 두 슬롯 사이 텍스트. 토큰 수가 고정이라 디코드가 오프셋으로 찾을 수 있다.\n'
  '    BBOX_MID = "; Subtask:"', None),

 ('    def tokenize_with_subtask(self, prompt: str, state: np.ndarray, subtask: str,\n'
  '                              mistake: bool | None = None):',
  '    def tokenize_with_subtask(self, prompt: str, state: np.ndarray, subtask: str,\n'
  '                              mistake: bool | None = None,\n'
  '                              bbox: "np.ndarray | None" = None):', None),

 ('        meta = "" if mistake is None else f" Mistake: {\'true\' if mistake else \'false\'};"\n'
  '        head = self._tokenizer.encode(f"State: {state_str};{meta} Subtask:", add_bos=True)',
  '        meta = "" if mistake is None else f" Mistake: {\'true\' if mistake else \'false\'};"\n'
  f'        # {MARK}: bbox 를 쓰면 Pen 슬롯이 Subtask 앞에 들어간다.\n'
  '        #   bbox = [ymin, xmin, ymax, xmax, visible] (0~1023 정수) 또는 None.\n'
  '        #   visible=0 이면 슬롯이 패딩이고 loss 에서 빠진다.\n'
  '        pad_id0 = self._tokenizer.pad_id() if self._tokenizer.pad_id() >= 0 else 0\n'
  '        if bbox is None:\n'
  '            head = self._tokenizer.encode(f"State: {state_str};{meta} Subtask:", add_bos=True)\n'
  '            bbox_body, mid = [], []\n'
  '        else:\n'
  '            head = self._tokenizer.encode(f"State: {state_str};{meta} Pen:", add_bos=True)\n'
  '            b = np.asarray(bbox).reshape(-1).astype(int)\n'
  '            if b.size >= 5 and b[4] > 0:\n'
  '                # <locNNNN> 는 사전에 이미 있다. id = 256000 + N 로 바로 쓴다.\n'
  '                base = self._tokenizer.piece_to_id("<loc0000>")\n'
  '                bbox_body = [base + int(min(max(v, 0), 1023)) for v in b[:4]]\n'
  '            else:\n'
  '                bbox_body = [pad_id0] * self.BBOX_SLOT\n'
  '            mid = self._tokenizer.encode(self.BBOX_MID)', None),

 ('        body = body + [pad_id] * (self.SUBTASK_SLOT - n_body)\n'
  '        tail = self._tokenizer.encode(";\\nAction: ")\n'
  '\n'
  '        tokens = head + body + tail\n'
  '        lo = len(head)',
  '        body = body + [pad_id] * (self.SUBTASK_SLOT - n_body)\n'
  '        tail = self._tokenizer.encode(";\\nAction: ")\n'
  '\n'
  f'        # {MARK}: head | [bbox 슬롯] | mid | [subtask 슬롯] | tail\n'
  '        tokens = head + bbox_body + mid + body + tail\n'
  '        lo_bbox = len(head)\n'
  '        lo = lo_bbox + len(bbox_body) + len(mid)', None),

 # 처음부터 **최종형** 을 내보낸다. 예전에는 중간형을 넣고 apply_bbox_fix 가
 # 덮어썼는데, 그러면 setup 재실행 때 이 패치가 중간형을 찾다 앵커 0 개로 죽는다.
 ('        loss = [lo <= i < lo + self.SUBTASK_SLOT for i in range(self._max_len)]',
  f'        # {MARK}: loss mask 는 **가시성과 무관하게** 두 슬롯을 덮는다.\n'
  '        #   이 마스크는 attention causal 블록 경계와 디코드 시작점으로도 쓰인다.\n'
  '        #   라벨이 없다고 그게 달라지면 추론에서 슬롯 위치가 밀린다.\n'
  '        #   감독 여부는 CE 쪽에서 `tgt == pad_id` 로 가린다.\n'
  '        loss = [(lo <= i < lo + self.SUBTASK_SLOT)\n'
  '                or (lo_bbox <= i < lo_bbox + len(bbox_body))\n'
  '                for i in range(self._max_len)]', None),
]

# ── 2. transforms.py ───────────────────────────────────────────────────────
TR = "src/openpi/transforms.py"
TR_INJECT = f'''@dataclasses.dataclass(frozen=True)
class InjectBBox(DataTransformFn):
    """{MARK}: (episode_index, frame_index) -> [ymin, xmin, ymax, xmax, visible].

    라벨은 pen_pose 와 고정 카메라로 **투영해서** 만든 GT 다 (탐지기 아님).
    labels_dir/ep{{episode:04d}}.npy 가 (T,5) int16 로 들어 있다.

    추론에는 repack 그룹이 돌지 않으므로 이 transform 도 돌지 않는다 —
    그때는 모델이 슬롯을 생성한다.
    """

    labels_dir: str = ""

    def __call__(self, data: DataDict) -> DataDict:
        if not self.labels_dir or "episode_index" not in data or "frame_index" not in data:
            return data
        ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        fr = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        p = pathlib.Path(self.labels_dir) / f"ep{{ep:04d}}.npy"
        if not p.exists():
            return data
        t = np.load(p, mmap_mode="r")
        if fr >= len(t):
            return data
        return {{**data, "bbox": np.asarray(t[fr], np.int32)}}


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):'''

TR_P = [
 ('@dataclasses.dataclass(frozen=True)\nclass TokenizePrompt(DataTransformFn):', TR_INJECT, None),
 ('    mistake_field: bool = False',
  '    mistake_field: bool = False\n'
  f'    # {MARK}: True 면 Pen 슬롯을 만든다. 학습·추론이 같아야 한다.\n'
  '    bbox_slot: bool = False', None),
 # 옛 설치에 남아 있는 bbox 블록을 **통째로 교체**한다. 안 하면 새 블록이
 # 그 아래에 또 붙어 `data.pop("bbox")` 가 두 번 돌고 **두 번째가 None 을 받아
 # GT 가 조용히 사라진다** (오류 없이 bbox 감독만 빠진다).
 ('''        # RSC_BBOX: 라벨이 없으면(추론) 빈 슬롯을 만들어 모델이 채우게 한다.
        bbox = data.pop("bbox", None)
        if bbox is None and self.bbox_slot:
            bbox = np.zeros(5, np.int32)
        subtask = data.pop("subtask", None)''',
  '        subtask = data.pop("subtask", None)', None),

 ('        subtask = data.pop("subtask", None)',
  f'        # {MARK}: 라벨이 없으면(추론) 빈 슬롯을 만들어 모델이 채우게 한다.\n'
  '        bbox = data.pop("bbox", None)\n'
  f'        # {MARK}: bbox 를 끄면 라벨이 들어와도 버린다 — 안 그러면 토크나이저는\n'
  '        #   슬롯을 만들고 모델은 없다고 가정해 CE 인덱스가 어긋난다.\n'
  '        if not self.bbox_slot:\n'
  '            bbox = None\n'
  '        elif bbox is None:\n'
  '            bbox = np.zeros(5, np.int32)\n'
  '        subtask = data.pop("subtask", None)', None),
 ('            tokens, token_masks, loss_mask = self.tokenizer.tokenize_with_subtask(\n'
  '                prompt, state, subtask, mistake=mistake)',
  '            tokens, token_masks, loss_mask = self.tokenizer.tokenize_with_subtask(\n'
  '                prompt, state, subtask, mistake=mistake, bbox=bbox)', None),
]

# ── 3. libero_policy.py ────────────────────────────────────────────────────
POL = "src/openpi/policies/libero_policy.py"
POL_P = [
 ('        return inputs',
  f'        # {MARK}: dict 를 새로 만들므로 명시하지 않으면 라벨이 사라진다.\n'
  '        if "bbox" in data:\n'
  '            inputs["bbox"] = data["bbox"]\n'
  '\n'
  '        return inputs', "class EmbodiChainInputs"),
]

# ── 4. config.py ───────────────────────────────────────────────────────────
CFG = "src/openpi/training/config.py"
CFG_P = [
 ('    mistake_dropout: float = 0.05',
  '    mistake_dropout: float = 0.05\n'
  f'    # {MARK}: 투영 GT 박스 라벨 디렉터리. 비면 bbox 슬롯을 안 만든다.\n'
  '    bbox_labels_dir: str = ""', None),
 ('''        if self.mistake_starts:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectMistake(starts=dict(self.mistake_starts),
                                                  dropout=self.mistake_dropout)],
            )''',
  '''        if self.mistake_starts:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectMistake(starts=dict(self.mistake_starts),
                                                  dropout=self.mistake_dropout)],
            )

        # ''' + MARK + ''': 학습 전용. 추론에는 repack 이 돌지 않아 모델이 슬롯을 생성한다.
        if self.bbox_labels_dir:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectBBox(labels_dir=self.bbox_labels_dir)],
            )''', None),
 ('                            mistake_field=_os.environ.get("PI05_MISTAKE", "0") == "1",',
  '                            mistake_field=_os.environ.get("PI05_MISTAKE", "0") == "1",\n'
  f'                            # {MARK}: 학습·추론에 같은 값이 들어가야 한다.\n'
  '                            bbox_slot=_os.environ.get("PI05_BBOX", "0") == "1",', None),
]


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    total = 0
    for rel, patches in ((TOK, TOK_P), (TR, TR_P), (POL, POL_P), (CFG, CFG_P)):
        p = root / rel
        s = p.read_text()
        n = 0
        for old, new, scope in patches:
            s, did = rep(s, old, new, scope=scope, path=rel)
            n += did
        # transforms.py 는 pathlib 이 필요하다
        if rel == TR and "import pathlib" not in s:
            s = s.replace("import dataclasses", "import dataclasses\nimport pathlib", 1)
        p.write_text(s)
        print(f"  {rel:46} {n}/{len(patches)} 적용")
        total += n
    print(f"BBOX-PATCH-DONE  {total} 곳")


if __name__ == "__main__":
    main()
