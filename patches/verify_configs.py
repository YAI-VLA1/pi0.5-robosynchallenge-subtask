#!/usr/bin/env python3
"""config 를 실제로 **로드**해 본다. 문자열 검사(verify_patches.py)의 보완이다.

리뷰 지적: --all 검증은 문자열 존재 검사라서, 연결이 잘못된 상태에서도
VERIFY-PATCHES OK 가 나온다. 실제로 그랬다 (DataConfig 에 assets 를 잘못 넣어
config.py 가 import 조차 안 되는데 문자열 검사는 통과했다).

여기서 보는 것
  · config.py 가 import 되는가 (문법·필드 이름)
  · 각 config 의 DataConfig 가 만들어지는가
  · norm_stats 가 실제로 **읽혔는가** (0 이면 경로가 틀린 것이다)
  · weight_loader 가 가리키는 경로가 존재하는가 (로컬 경로일 때만)

import 순서는 scripts/train.py 를 따른다 (config 를 먼저 올리면 segfault).
"""
import pathlib, sys
import etils.epath as epath  # noqa: F401
import flax.nnx as nnx  # noqa: F401
import jax  # noqa: F401
import openpi.models.model as _model  # noqa: F401
import openpi.training.config as _config


def main():
    names = sys.argv[1:]
    if not names:
        sys.exit("사용법: verify_configs.py <config 이름> ...")
    bad = 0
    for n in names:
        try:
            c = _config.get_config(n)
            dc = c.data.create(c.assets_dirs, c.model)
        except Exception as e:
            print(f"  ✗  {n}\n       {type(e).__name__}: {e}")
            bad += 1
            continue
        ns = len(dc.norm_stats or {})
        wl = getattr(c.weight_loader, "params_path", "")
        wl_ok = True
        if wl and not str(wl).startswith("gs://"):
            wl_ok = pathlib.Path(wl).exists()
        marks = []
        if ns == 0:
            marks.append("norm_stats 없음")
        if not wl_ok:
            marks.append(f"weight_loader 경로 없음 {wl}")
        if marks:
            print(f"  ✗  {n:58} " + " · ".join(marks))
            bad += 1
        else:
            src = "base" if str(wl).startswith("gs://") else pathlib.Path(wl).parent.name
            print(f"  OK {n:58} steps={c.num_train_steps:>6} norm_stats={ns} from={src}")
    print()
    if bad:
        sys.exit(f"VERIFY-CONFIGS FAIL — {bad} 개")
    print("VERIFY-CONFIGS OK")


if __name__ == "__main__":
    main()
