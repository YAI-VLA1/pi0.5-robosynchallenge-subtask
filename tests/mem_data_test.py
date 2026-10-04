#!/usr/bin/env python3
"""데이터 로더가 과거 프레임과 is_pad 를 제대로 주는지 확인한다.

  · 이미지가 (T,C,H,W) 로 쌓여 나오는가
  · 에피소드 시작부에서 과거가 패딩되고 is_pad 로 표시되는가
  · 에피소드 중간에서는 전부 valid 인가
  · 1초 간격이 실제 프레임 간격 25 와 맞는가 (fps 25)
"""
import os, sys
import numpy as np

T = int(os.environ.get("PI05_MEM_FRAMES", "6"))
import openpi.training.config as _config
import openpi.training.data_loader as _dl

cfg = _config.get_config(os.environ.get("MEM_CFG", "pi05_robosyn_items_handover_lora_cos82k"))
ds = _dl.create_torch_dataset(cfg.data.create(cfg.assets_dirs, cfg.model), cfg.model.action_horizon, cfg.model)
# prompt_from_task 가 켜져 있으면 TransformedDataset 로 감싸여 온다
base = ds
while not hasattr(base, "meta") and hasattr(base, "_dataset"):
    base = base._dataset
print(f"에피소드 {base.num_episodes} · 프레임 {base.num_frames} · fps {base.meta.fps}", flush=True)

img_keys = [k for k in base.meta.features if k.startswith("observation.images.")]
print(f"이미지 키 {img_keys}\n", flush=True)

def show(i, label):
    s = ds[i]
    k = img_keys[0]
    v = s[k]
    pad = s.get(f"{k}_is_pad")
    print(f"[{label}] idx {i}")
    print(f"   {k}  shape {tuple(v.shape)}")
    if pad is not None:
        p = np.asarray(pad).ravel()
        print(f"   is_pad {p.astype(int).tolist()}  (True = 과거가 없어 패딩)")
    # 프레임이 실제로 다른가 (같은 프레임을 복제한 게 아닌지)
    a = np.asarray(v, dtype=np.float32)
    if a.ndim == 4 and a.shape[0] > 1:
        d = [float(np.abs(a[j] - a[-1]).mean()) for j in range(a.shape[0])]
        print(f"   현재와의 평균차 {['%.4f' % x for x in d]}")
    return s

show(0, "에피소드 시작 (과거 없음)")
print()
show(200, "에피소드 중간")
print("\nMEM-DATA-TEST-DONE")
