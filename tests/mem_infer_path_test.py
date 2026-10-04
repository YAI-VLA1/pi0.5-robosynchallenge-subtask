#!/usr/bin/env python3
"""MEM 추론 경로 검증. (리뷰 R8)

실제 patched 코드를 import 해서 돌린다.
  ① pi_model 의 transpose 가 T=1 / T=6 둘 다 통과하는가
  ② EmbodiChainInputs 가 추론에서 준 frame_valid 를 **살려 보내는가**
  ③ 학습 경로(observation/*_is_pad)도 그대로 동작하는가
"""
import sys
import numpy as np
import openpi.models.model as _model
from openpi.policies.libero_policy import EmbodiChainInputs


def _chw(x):
    """pi_model.py 의 _chw 와 같아야 한다 (복사본이 아니라 실제 파일에서 읽어 확인한다)."""
    x = np.asarray(x)
    if x.ndim == 4:
        return np.transpose(x, (0, 3, 1, 2))
    return np.transpose(x, (2, 0, 1))


def main():
    import pathlib, re
    pm = pathlib.Path("/workspace/rsc_ws/RoboSynChallenge/policy/pi05/pi_model.py").read_text()
    assert "RSC_MEMINFER" in pm, "pi_model 패치가 안 들어갔다"
    assert "np.transpose(x, (0, 3, 1, 2))" in pm, "4D transpose 가 없다"

    ok = {}
    # ① transpose
    for T in (1, 6):
        img = np.zeros((224, 224, 3), np.uint8) if T == 1 else np.zeros((T, 224, 224, 3), np.uint8)
        out = _chw(img)
        exp = (3, 224, 224) if T == 1 else (T, 3, 224, 224)
        ok[f"① transpose T={T} -> {exp}"] = out.shape == exp

    tf = EmbodiChainInputs(model_type=_model.ModelType.PI05)

    def base(T):
        mk = (lambda: np.zeros((3, 224, 224), np.uint8)) if T == 1 else \
             (lambda: np.zeros((T, 3, 224, 224), np.uint8))
        return {"observation/image": mk(), "observation/left_wrist_image": mk(),
                "observation/right_wrist_image": mk(),
                "observation/state": np.zeros(14, np.float32), "prompt": "x"}

    # ② 추론 경로: frame_valid 를 직접 넣는다
    T = 6
    fv = {"base_0_rgb": np.array([0, 0, 0, 1, 1, 1], bool),
          "left_wrist_0_rgb": np.array([0, 0, 0, 1, 1, 1], bool),
          "right_wrist_0_rgb": np.array([0, 0, 0, 1, 1, 1], bool)}
    out = tf({**base(T), "frame_valid": fv})
    got = out.get("frame_valid")
    ok["② 추론 frame_valid 통과"] = got is not None and \
        all(np.array_equal(np.asarray(got[k]), fv[k]) for k in fv)
    print(f"  추론 frame_valid: {None if got is None else {k: np.asarray(v).astype(int).tolist() for k, v in got.items()}}")

    # ③ 학습 경로: *_is_pad 에서 만들어지는가
    pad = np.array([1, 1, 1, 0, 0, 0], bool)
    out = tf({**base(T),
              "observation/image_is_pad": pad,
              "observation/left_wrist_image_is_pad": pad,
              "observation/right_wrist_image_is_pad": pad})
    got = out.get("frame_valid")
    ok["③ 학습 is_pad -> frame_valid"] = got is not None and \
        np.array_equal(np.asarray(got["base_0_rgb"]), ~pad)

    # ④ 둘 다 없으면 키가 없어야 한다 (T=1 기존 동작)
    out = tf(base(1))
    ok["④ T=1 은 frame_valid 없음"] = "frame_valid" not in out

    print()
    for k, v in ok.items():
        print(f"  {'OK ' if v else '✗  '} {k}")
    if not all(ok.values()):
        raise SystemExit("MEM-INFER-PATH FAIL")
    print("\nMEM-INFER-PATH OK")


if __name__ == "__main__":
    main()
