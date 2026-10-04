#!/usr/bin/env python3
"""MEM 의 추론 비용: T=1 vs T=6, 카메라 3대 기준.

점수의 5% 가 추론 효율(ACT 베이스라인 대비)이라 이 수치가 채택 여부를 가른다.
GPU 를 수집과 나눠 쓰는 중이면 절대값은 오염되지만, **T=1 과 T=6 을 번갈아 재서**
비율은 유지되게 한다.

정책 호출 1회 = 카메라 3대 인코딩. 현재 정책 전체는 호출당 약 1,870 ms.
"""
import os, time
import numpy as np, jax, jax.numpy as jnp
import flax.traverse_util as tu, etils.epath as epath
import openpi.models.model as _model
from openpi.models.siglip import Encoder

CKPT = os.environ.get("MEM_CKPT", "/workspace/ckpt_dl/subtask/checkpoints/81999")
DEPTH, HEADS, MLP, D = 27, 16, 4304, 1152
N = 256          # 실제 패치 수 (224x224, patch 14)
CAMS = int(os.environ.get("CAMS", "1"))   # 메모리 때문에 1대로 재고 3배 환산한다
REPS = int(os.environ.get("REPS", "12"))
POLICY_MS = 1870.0

dev = jax.devices()[0]
print(f"장치 {dev}", flush=True)
flat = tu.flatten_dict(_model.restore_params(epath.Path(CKPT) / "params", dtype=jnp.float32))
pref = ("PaliGemma", "img", "Transformer")
enc = tu.unflatten_dict({k[len(pref):]: v for k, v in flat.items() if k[:len(pref)] == pref})
params = {"params": enc}

def make(T):
    m = Encoder(depth=DEPTH, mlp_dim=MLP, num_heads=HEADS, dropout=0.0,
                scan=True, dtype_mm="float32", num_frames=T)
    @jax.jit
    def f(x, fv):
        y, _ = m.apply(params, x, True, fv)
        return y
    return f

def bench(T):
    f = make(T)
    x = jnp.asarray(np.random.randn(CAMS * T, N, D), dtype=jnp.float32)
    fv = jnp.ones((CAMS, T), dtype=bool) if T > 1 else None
    f(x, fv).block_until_ready()          # 컴파일
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        f(x, fv).block_until_ready()
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)

# T=1 과 T=6 을 번갈아 재서 GPU 경합이 양쪽에 같이 걸리게 한다
r = {1: [], 6: []}
f1, f6 = make(1), make(6)
x1 = jnp.asarray(np.random.randn(CAMS * 1, N, D), dtype=jnp.float32)
x6 = jnp.asarray(np.random.randn(CAMS * 6, N, D), dtype=jnp.float32)
fv6 = jnp.ones((CAMS, 6), dtype=bool)
f1(x1, None).block_until_ready(); f6(x6, fv6).block_until_ready()
for _ in range(REPS):
    t0 = time.perf_counter(); f1(x1, None).block_until_ready(); r[1].append((time.perf_counter()-t0)*1000)
    t0 = time.perf_counter(); f6(x6, fv6).block_until_ready(); r[6].append((time.perf_counter()-t0)*1000)

a, b = np.array(r[1]), np.array(r[6])
SCALE = 3 / CAMS      # 정책은 카메라 3대를 쓴다
print(f"\n카메라 {CAMS}대로 측정 · 패치 {N} · depth {DEPTH} · {REPS}회 (3대 환산 x{SCALE:.0f})")
print(f"  T=1   {np.median(a):7.1f} ms/대 -> 3대 {np.median(a)*SCALE:7.1f} ms")
print(f"  T=6   {np.median(b):7.1f} ms/대 -> 3대 {np.median(b)*SCALE:7.1f} ms")
print(f"  배율  {np.median(b)/np.median(a):.2f}x   증가분 {np.median(b)-np.median(a):+.1f} ms")
print(f"\n정책 호출 {POLICY_MS:.0f} ms 기준")
print(f"  T=1 비전 비중 {100*np.median(a)*SCALE/POLICY_MS:.1f}%")
print(f"  T=6 추가 비용 {100*(np.median(b)-np.median(a))*SCALE/POLICY_MS:+.1f}%")
print("MEM-COST-DONE")
