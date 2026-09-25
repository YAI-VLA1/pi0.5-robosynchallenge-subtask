#!/usr/bin/env python3
"""손실에서 패딩 축을 제외한다 (pi05, JAX + PyTorch 양쪽).

배경
  pi05 체크포인트가 32차원 action 으로 사전학습돼 있어 `action_dim` 은 32 로 고정이다.
  우리 로봇은 14축이라 `transforms.PadStatesAndActions` 가 뒤 18축을 0 으로 채운다.
  손실은 `jnp.mean(..., axis=-1)` 로 32축을 마스크 없이 평균낸다.

이 패치가 실제로 바꾸는 것
  ★ 학습 결과는 바뀌지 않는다. 패딩 축을 빼는 것은 손실 전체를 32/14 = 2.29배로
    균일 스케일링하는 것과 같고, AdamW 는 기울기의 균일 스케일에 불변이다.
  ○ 바뀌는 것은 로그에 찍히는 loss 값이다. 지금은 상수·패딩 축 수에 따라 최대 5배까지
    희석돼서 태스크가 다른 런끼리 loss 를 비교할 수 없다. 마스킹하면 비교 가능해진다.

  (2026-09-25 측정: 축별 손실 몫은 1/32 균등이 아니라 타깃 분산에 비례한다.
   items_handover 좌그리퍼는 14축 중 24% 로 두 번째로 큰 축이었다.
   "그리퍼가 과소가중이라 못 배운다"는 가설은 이 패치로 해결되지 않는다 — 틀린 가설이었다.)

패딩이 뒤쪽 연속 구간이므로 마스크는 `[..., :d]` 슬라이스로 충분하다
(`transforms.pad_to_dim` 은 뒤에 붙인다).

축 수는 환경변수 `PI05_REAL_ACTION_DIM` 로 바꾼다. 기본 14.
`PI05_REAL_ACTION_DIM=0` 이면 마스킹하지 않는다(원래 동작).

사용:  python3 apply_loss_mask.py <repo 루트>        # 적용
       python3 apply_loss_mask.py <repo 루트> --check # 적용 여부만 확인
"""
from __future__ import annotations

import pathlib
import sys

MARK = "RSC_LOSS_MASK"

JAX_FILE = "policy/pi05/src/openpi/models/pi0.py"
JAX_ANCHOR = """        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)"""
JAX_NEW = """        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # {mark}: 패딩 축을 손실에서 제외한다. pad_to_dim 이 뒤에 붙이므로 앞 d 축만 쓴다.
        d = _RSC_REAL_ACTION_DIM
        if 0 < d < v_t.shape[-1]:
            return jnp.mean(jnp.square(v_t[..., :d] - u_t[..., :d]), axis=-1)
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)""".format(mark=MARK)

JAX_HEADER_ANCHOR = "class Pi0(_model.BaseModel):"
JAX_HEADER = """# {mark}: 실제 로봇 축 수. 나머지(32 - d)는 0 패딩이라 손실에서 뺀다. 0 이면 마스킹 안 함.
_RSC_REAL_ACTION_DIM = int(_os.environ.get("PI05_REAL_ACTION_DIM", "14"))
logging.info("{mark}: 손실 마스킹 축 수 = %d (0 이면 비활성)", _RSC_REAL_ACTION_DIM)


""".format(mark=MARK)

PT_FILE = "policy/pi05/src/openpi/models_pytorch/pi0_pytorch.py"
PT_ANCHOR = """        return F.mse_loss(u_t, v_t, reduction="none")"""
PT_NEW = """        # {mark}: JAX 경로(models/pi0.py)와 같은 마스킹. 패딩 축은 뒤에 붙어 있다.
        d = _RSC_REAL_ACTION_DIM
        if 0 < d < v_t.shape[-1]:
            return F.mse_loss(u_t[..., :d], v_t[..., :d], reduction="none")
        return F.mse_loss(u_t, v_t, reduction="none")""".format(mark=MARK)
PT_HEADER_ANCHOR = "class PI0Pytorch(nn.Module):"
PT_HEADER = """# {mark}: models/pi0.py 와 같은 값을 쓴다.
_RSC_REAL_ACTION_DIM = int(_os.environ.get("PI05_REAL_ACTION_DIM", "14"))


""".format(mark=MARK)


def patch(root: pathlib.Path, rel: str, header_anchor: str, header: str,
          anchor: str, new: str, check: bool) -> bool:
    p = root / rel
    if not p.exists():
        print(f"  {rel}: 파일 없음 — 건너뜀")
        return False
    s = p.read_text()
    if MARK in s:
        print(f"  {rel}: 이미 적용됨")
        return True
    if anchor not in s:
        print(f"  {rel}: ✗ 앵커를 못 찾음 — 업스트림이 바뀌었다. 직접 확인할 것")
        return False
    if check:
        print(f"  {rel}: 미적용 (적용 가능)")
        return False
    if "import os as _os" not in s:
        # 기존 import 블록 바로 뒤에 붙인다.
        first = s.index("import ")
        s = s[:first] + "import os as _os\n" + s[first:]
    if header_anchor not in s:
        print(f"  {rel}: ✗ 헤더 앵커를 못 찾음")
        return False
    s = s.replace(header_anchor, header + header_anchor, 1)
    s = s.replace(anchor, new, 1)
    p.write_text(s)
    print(f"  {rel}: ✓ 적용")
    return True


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    check = "--check" in sys.argv
    print(f"{'확인' if check else '적용'}: {root}")
    a = patch(root, JAX_FILE, JAX_HEADER_ANCHOR, JAX_HEADER, JAX_ANCHOR, JAX_NEW, check)
    b = patch(root, PT_FILE, PT_HEADER_ANCHOR, PT_HEADER, PT_ANCHOR, PT_NEW, check)
    if check:
        sys.exit(0 if (a and b) else 1)
    if not (a and b):
        sys.exit("일부 파일에 적용 실패 — 위 메시지 확인")
    print("\n적용 완료. 확인:")
    print(f"  grep -n {MARK} {root}/{JAX_FILE}")
    print("끄려면 학습 실행 시 PI05_REAL_ACTION_DIM=0")


if __name__ == "__main__":
    main()
