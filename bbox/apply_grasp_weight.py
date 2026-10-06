#!/usr/bin/env python3
"""집기 구간의 flow-matching 손실에 가중을 준다.

왜
  성패를 가르는 '그리퍼 닫기' 가 데이터의 **4.9%** 뿐이다 (327 중 16 프레임).
  나머지 95% 는 이동·운반이고 그건 이미 잘한다. 전형적인 클래스 불균형이다.

왜 오버샘플링이 아니라 가중인가
  샘플을 중복시키면 그 프레임의 **이미지까지** 반복 학습돼 과적합이 빨리 온다.
  가중은 데이터 분포를 그대로 두고 기울기 크기만 바꾼다. 되돌리기도 쉽다.

범위
  subtask_ends 로 구간을 안다. 기본값은 '그리퍼 닫기' 전후를 덮는
  [lower onto pen 시작, lift 끝) 이다 — 닫는 순간만 좁게 잡으면 그 앞의
  접근 궤적이 안 바뀌어 효과가 적다.

켜기: PI05_GRASP_W=3.0  (1.0 이면 꺼진 것과 같다)
"""
import pathlib, sys

MARK = "RSC_GRASPW"

# ── ① transforms.py — frame_index -> 가중치 ────────────────────────────────
TR = "src/openpi/transforms.py"
TR_A = '''@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):'''
TR_B = f'''@dataclasses.dataclass(frozen=True)
class InjectGraspWeight(DataTransformFn):
    """{MARK}: frame_index -> 손실 가중치.

    집기 구간([lo, hi) 프레임)에만 weight 를 주고 나머지는 1.0 이다.
    sim 에피소드는 스크립트 재생이라 구간이 프레임 번호로 결정된다.

    추론에는 repack 그룹이 돌지 않으므로 이 transform 도 돌지 않는다.
    """

    lo: int = 0
    hi: int = 0
    weight: float = 1.0

    def __call__(self, data: DataDict) -> DataDict:
        if self.weight == 1.0 or self.hi <= self.lo or "frame_index" not in data:
            return data
        f = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        w = self.weight if self.lo <= f < self.hi else 1.0
        return {{**data, "loss_weight": np.float32(w)}}


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):'''

# ── ② libero_policy.py — 통과 ──────────────────────────────────────────────
POL = "src/openpi/policies/libero_policy.py"
POL_A = '''        return inputs'''
POL_B = f'''        # {MARK}: dict 를 새로 만들므로 명시하지 않으면 가중치가 사라진다.
        if "loss_weight" in data:
            inputs["loss_weight"] = data["loss_weight"]

        return inputs'''

# ── ③ model.py — Observation 필드 ─────────────────────────────────────────
MOD = "src/openpi/models/model.py"
MOD_A = '''    frame_valid: dict[str, at.Bool[ArrayT, "*fb t"]] | None = None'''
MOD_B = f'''    frame_valid: dict[str, at.Bool[ArrayT, "*fb t"]] | None = None
    # {MARK}: 샘플별 손실 가중치. 집기 구간처럼 드문 결정적 순간을 키운다.
    loss_weight: at.Float[ArrayT, "*wb"] | None = None'''
MOD_A2 = '''            frame_valid=data.get("frame_valid"),'''
MOD_B2 = '''            frame_valid=data.get("frame_valid"),
            loss_weight=data.get("loss_weight"),'''

# ── ④ pi0.py — flow 손실에 곱한다 ─────────────────────────────────────────
PI0 = "src/openpi/models/pi0.py"
PI0_A = '''        ce, acc, acc_first = self._rsc_subtask_ce(observation, prefix_out)
        total = flow + _RSC_SUBTASK_W * ce[:, None]'''
PI0_B = f'''        ce, acc, acc_first = self._rsc_subtask_ce(observation, prefix_out)
        total = flow + _RSC_SUBTASK_W * ce[:, None]
        # {MARK}: 집기 같은 드문 결정적 구간을 키운다. 없으면 1.0 이다.
        if observation.loss_weight is not None:
            total = total * observation.loss_weight[:, None]'''

# ── ⑤ config.py — 배선 ────────────────────────────────────────────────────
CFG = "src/openpi/training/config.py"
CFG_A = '''    bbox_labels_dir: str = ""'''
CFG_B = f'''    bbox_labels_dir: str = ""
    # {MARK}: 집기 구간 [lo, hi) 과 가중치. weight 는 PI05_GRASP_W 로 덮어쓴다.
    grasp_window: tuple[int, int] = (0, 0)'''
CFG_A2 = '''        if self.bbox_labels_dir:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectBBox(labels_dir=self.bbox_labels_dir)],
            )'''
CFG_B2 = f'''        if self.bbox_labels_dir:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectBBox(labels_dir=self.bbox_labels_dir)],
            )

        # {MARK}: 학습 전용. 추론에는 repack 이 돌지 않아 가중치가 안 붙는다(=1.0).
        _gw = float(_os.environ.get("PI05_GRASP_W", "1.0"))
        if self.grasp_window[1] > self.grasp_window[0] and _gw != 1.0:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectGraspWeight(
                    lo=self.grasp_window[0], hi=self.grasp_window[1], weight=_gw)],
            )'''


def rep(root, rel, pairs, scope=None):
    """scope 가 있으면 그 클래스 본문 안에서만 찾는다.

    `return inputs` 는 LiberoInputs 와 EmbodiChainInputs 양쪽에 똑같이 있다.
    범위를 안 좁히면 앵커가 2번 나와 멈춘다.
    """
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
            raise SystemExit(f"★ {rel}: 앵커가 {n} 번 (1 이어야 함)\n{a.splitlines()[0][:80]}")
    for a, b in pairs:
        body = body.replace(a, b, 1)
    p.write_text(s[:lo] + body + s[hi:])
    print(f"  {rel}: ✓ 적용")


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    rep(root, TR,  [(TR_A, TR_B)])
    rep(root, POL, [(POL_A, POL_B)], scope="class EmbodiChainInputs")
    rep(root, MOD, [(MOD_A, MOD_B), (MOD_A2, MOD_B2)])
    rep(root, PI0, [(PI0_A, PI0_B)])
    rep(root, CFG, [(CFG_A, CFG_B), (CFG_A2, CFG_B2)])
    print("GRASP-WEIGHT-DONE")


if __name__ == "__main__":
    main()
