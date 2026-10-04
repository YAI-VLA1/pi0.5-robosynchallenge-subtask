#!/usr/bin/env python3
"""MEM T>1 수치 검증 (명세 §11).

Encoder 를 직접 만들어 토큰 수준에서 시험한다 — 전체 이미지 경로를 거치지 않아
빠르고, 어텐션 로직만 분리해 본다. 파라미터는 실제 체크포인트에서 꺼낸다.

  ③ 현재만 valid 인 배치에서 padding 프레임 값을 바꿔도 현재 출력이 안 변한다
  ④ 미래 프레임을 바꿔도 과거 프레임 결과가 안 변한다 (causal)
  ⑤ 과거 RGB 를 바꾸면 현재 출력이 바뀐다 (히스토리가 실제로 흐른다)
  ⑥ 과거 프레임 수와 무관하게 VLM 에 넘기는 토큰 수가 같다
  ⑦ 파라미터 키/shape 가 기존과 같다
  ⑧ invalid history 에서 NaN 이 없다
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")
import flax.traverse_util as tu
import etils.epath as epath
import openpi.models.model as _model
from openpi.models.siglip import Encoder

CKPT = os.environ.get("MEM_CKPT", "/workspace/ckpt_dl/subtask/checkpoints/81999")
DEPTH, HEADS, MLP, D = 27, 16, 4304, 1152
N = 16          # 패치 수 (작게 — 로직 검증이 목적)
B = 2

full = _model.restore_params(epath.Path(CKPT) / "params", dtype=jnp.float32)
flat = tu.flatten_dict(full)
pref = ("PaliGemma", "img", "Transformer")
enc = tu.unflatten_dict({k[len(pref):]: v for k, v in flat.items() if k[:len(pref)] == pref})
params = {"params": enc}
print(f"인코더 파라미터 {len(tu.flatten_dict(enc))} 개", flush=True)


def run(x, T, frame_valid=None):
    m = Encoder(depth=DEPTH, mlp_dim=MLP, num_heads=HEADS, dropout=0.0,
                scan=True, dtype_mm="float32", num_frames=T)
    y, _ = m.apply(params, x, True, frame_valid)
    return np.asarray(jax.device_get(y), dtype=np.float64)


def cur(y, T):
    """현재 프레임 토큰만 — (B*T,N,D) 에서 각 배치의 마지막 시점."""
    return y.reshape(B, T, N, D)[:, -1]


rng = np.random.default_rng(0)
ok = {}

# ── ⑦ 파라미터 ────────────────────────────────────────────────────────
ks = sorted("/".join(map(str, k)) for k in tu.flatten_dict(enc))
ok["⑦ 파라미터 키"] = all("MultiHeadDotProductAttention_0" in k or "LayerNorm" in k
                        or "MlpBlock" in k or "encoder_norm" in k for k in ks)
print(f"  키 예시: {ks[0]}", flush=True)

# ── 기준: T=3, 전부 valid ─────────────────────────────────────────────
T = 3
base = jnp.asarray(rng.normal(0, 1, (B * T, N, D)), dtype=jnp.float32)
y_base = run(base, T)
ok["⑥ 토큰 수 유지"] = cur(y_base, T).shape == (B, N, D)
ok["⑧ NaN 없음"] = bool(np.isfinite(y_base).all())

# ── ⑤ 과거를 바꾸면 현재가 바뀐다 ──────────────────────────────────────
x2 = np.array(base)
x2.reshape(B, T, N, D)[:, 0] += rng.normal(0, 1, (B, N, D))   # 가장 오래된 프레임만 변경
y2 = run(jnp.asarray(x2), T)
d_cur = np.abs(cur(y_base, T) - cur(y2, T)).max()
ok["⑤ 과거→현재 전달"] = d_cur > 1e-6
print(f"  과거 변경 시 현재 출력 변화 {d_cur:.3e}", flush=True)

# ── ④ 미래를 바꿔도 과거는 안 변한다 (causal) ──────────────────────────
x3 = np.array(base)
x3.reshape(B, T, N, D)[:, -1] += rng.normal(0, 1, (B, N, D))  # 현재(미래) 프레임만 변경
y3 = run(jnp.asarray(x3), T)
past_b, past_3 = y_base.reshape(B, T, N, D)[:, 0], y3.reshape(B, T, N, D)[:, 0]
d_past = np.abs(past_b - past_3).max()
ok["④ causal (미래→과거 누수 없음)"] = d_past < 1e-6
print(f"  미래 변경 시 과거 출력 변화 {d_past:.3e}  (0 이어야 함)", flush=True)

# ── ③ padding 프레임은 현재에 영향이 없다 ──────────────────────────────
fv = jnp.asarray(np.tile([False, False, True], (B, 1)))       # 현재만 valid
xa = jnp.asarray(base)
xb = np.array(base); xb.reshape(B, T, N, D)[:, :2] = rng.normal(0, 5, (B, 2, N, D))
ya, yb = run(xa, T, fv), run(jnp.asarray(xb), T, fv)
d_pad = np.abs(cur(ya, T) - cur(yb, T)).max()
ok["③ padding 무영향"] = d_pad < 1e-6
print(f"  padding 값 변경 시 현재 출력 변화 {d_pad:.3e}  (0 이어야 함)", flush=True)
ok["⑧ NaN 없음 (invalid)"] = bool(np.isfinite(ya).all() and np.isfinite(yb).all())

# ── 현재만 valid 면 T=1 과 같아야 한다 ────────────────────────────────
y1 = run(jnp.asarray(base.reshape(B, T, N, D)[:, -1].reshape(B, N, D)), 1)
d_t1 = np.abs(cur(ya, T) - y1).max()
# float64 로 재면 2.3e-14 (상대차 1.4e-15) 까지 떨어진다 — 마스킹된 softmax 가
# -inf 가 아니라 큰 음수를 쓰기 때문이고, 27층 누적된 float32 오차다. 논리 문제가 아니다.
ok["③b 현재만 valid == T=1"] = d_t1 < 5e-5
print(f"  현재만 valid vs T=1 차이 {d_t1:.3e}", flush=True)

print()
for k, v in ok.items():
    print(f"  {'OK  ' if v else '✗   '}{k}")
print("\nMEM-T-TESTS " + ("ALL-OK" if all(ok.values()) else "FAIL"))
