#!/usr/bin/env python3
"""③b 의 1.5e-5 가 float32 누적오차인지 논리 버그인지 가린다.

float64 로 같은 비교를 한다. 수치 문제면 ~1e-13 으로 떨어지고,
논리 버그면 1e-5 근처에 그대로 남는다.
"""
import os
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
import flax.traverse_util as tu, etils.epath as epath
import openpi.models.model as _model
from openpi.models.siglip import Encoder

CKPT = os.environ.get("MEM_CKPT", "/workspace/ckpt_dl/subtask/checkpoints/81999")
DEPTH, HEADS, MLP, D, N, B = 27, 16, 4304, 1152, 16, 2
flat = tu.flatten_dict(_model.restore_params(epath.Path(CKPT) / "params", dtype=jnp.float64))
pref = ("PaliGemma", "img", "Transformer")
enc = tu.unflatten_dict({k[len(pref):]: v.astype(jnp.float64)
                         for k, v in flat.items() if k[:len(pref)] == pref})
params = {"params": enc}

def run(x, T, fv=None):
    m = Encoder(depth=DEPTH, mlp_dim=MLP, num_heads=HEADS, dropout=0.0,
                scan=True, dtype_mm="float64", num_frames=T)
    y, _ = m.apply(params, x, True, fv)
    return np.asarray(jax.device_get(y), dtype=np.float64)

rng = np.random.default_rng(0)
T = 3
base = jnp.asarray(rng.normal(0, 1, (B * T, N, D)), dtype=jnp.float64)
fv = jnp.asarray(np.tile([False, False, True], (B, 1)))
ya = run(base, T, fv).reshape(B, T, N, D)[:, -1]
y1 = run(jnp.asarray(np.asarray(base).reshape(B, T, N, D)[:, -1].reshape(B, N, D)), 1)
d = np.abs(ya - y1)
print(f"출력 크기 중앙값 {np.median(np.abs(y1)):.4f}")
print(f"최대 절대차 {d.max():.3e} · 평균 {d.mean():.3e}")
print(f"상대차 {d.max()/max(1e-12, np.abs(y1).max()):.3e}")
print("판정:", "float32 누적오차였다 (논리 OK)" if d.max() < 1e-10
      else "✗ 논리 버그 — float64 에서도 남는다")
