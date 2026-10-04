#!/usr/bin/env python3
"""전체 이미지 경로로 MEM 을 시험한다 — 실제 모델의 비전 타워를 그대로 쓴다.

토큰 수준 테스트(mem_t_tests.py)가 못 덮는 부분을 본다:
  · (B,T,H,W,3) 입력 처리와 프레임을 배치로 펴는 reshape
  · 인코더 출력에서 과거 토큰을 버려 (B,N,D) 를 유지하는지
  · 패치 추출·공간 posemb 가 프레임마다 제대로 도는지
"""
import os
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")
import etils.epath as epath
import openpi.models.model as _model
import openpi.training.config as _config

CFG  = os.environ.get("MEM_CFG",  "pi05_robosyn_items_handover_lora_cos82k")
CKPT = os.environ.get("MEM_CKPT", "/workspace/ckpt_dl/subtask/checkpoints/81999")
T    = int(os.environ.get("T", "3"))

params = _model.restore_params(epath.Path(CKPT) / "params", dtype=jnp.float32)
model  = _config.get_config(CFG).model.load(params)
img    = model.PaliGemma.img
inner  = img.module
print(f"비전 타워 width {inner.width} depth {inner.depth} · T={T}", flush=True)

rng = np.random.default_rng(0)
B, H, W = 1, 224, 224
frames = rng.uniform(-1, 1, (B, T, H, W, 3)).astype(np.float32)

def call(x, fv=None, nf=1):
    object.__setattr__(inner, "num_frames", nf)
    y, _ = img(jnp.asarray(x), train=False) if fv is None and nf == 1 else \
           img(jnp.asarray(x), train=False, frame_valid=fv)
    return np.asarray(jax.device_get(y), np.float64)

ok = {}
y1 = call(frames[:, -1], nf=1)
print(f"T=1 출력 {y1.shape}", flush=True)

fv_cur = jnp.asarray(np.array([[False] * (T - 1) + [True]]))
yc = call(frames, fv_cur, nf=T)
ok["토큰 수 유지"] = yc.shape == y1.shape
print(f"T={T} 출력 {yc.shape}", flush=True)
d = np.abs(yc - y1).max()
ok["현재만 valid == T=1"] = d < 5e-4
print(f"  현재만 valid vs T=1 차이 {d:.3e}", flush=True)

fv_all = jnp.ones((B, T), dtype=bool)
ya = call(frames, fv_all, nf=T)
f2 = frames.copy(); f2[:, 0] = rng.uniform(-1, 1, (B, H, W, 3))
yb = call(f2, fv_all, nf=T)
dh = np.abs(ya - yb).max()
ok["과거→현재 전달"] = dh > 1e-5
print(f"  과거 이미지 변경 시 현재 출력 변화 {dh:.3e}", flush=True)

object.__setattr__(inner, "num_frames", 1)
print()
for k, v in ok.items():
    print(f"  {'OK  ' if v else '✗   '}{k}")
print("\nMEM-IMAGE-PATH " + ("ALL-OK" if all(ok.values()) else "FAIL"))
