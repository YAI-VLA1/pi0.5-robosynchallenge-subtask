#!/usr/bin/env python3
"""데이터 로더 -> 변환 파이프라인 -> 모델 입력까지 히스토리가 살아서 가는지 본다."""
import os
import numpy as np
import openpi.training.config as _config
import openpi.training.data_loader as _dl

T = int(os.environ.get("PI05_MEM_FRAMES", "6"))
cfg = _config.get_config(os.environ.get("MEM_CFG", "pi05_robosyn_items_handover_lora_cos82k"))
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
ds = _dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
print(f"원본 샘플 키 수 {len(ds[0])}", flush=True)
raw = ds[0]
for k in sorted(raw):
    if "image" in k:
        v = np.asarray(raw[k])
        print(f"  {k:46} {tuple(v.shape)} {v.dtype}")

print("\n--- 변환 파이프라인 통과 후 ---", flush=True)
tds = _dl.transform_dataset(ds, dc, skip_norm_stats=True)
s = tds[0]
for k in sorted(s):
    v = s[k]
    if isinstance(v, dict):
        for kk, vv in v.items():
            a = np.asarray(vv)
            print(f"  {k}/{kk:34} {tuple(a.shape)} {a.dtype}")
    else:
        a = np.asarray(v)
        print(f"  {k:46} {tuple(a.shape)} {a.dtype}")
print("\nMEM-PIPELINE-DONE")
