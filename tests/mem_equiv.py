#!/usr/bin/env python3
"""비전 타워 출력을 고정 입력으로 뽑아 저장/비교한다.

MEM 패치가 K=1 에서 기존과 **수치적으로 동일**해야 한다.
다르면 구현이 틀린 것이다 — 지난번 양방향 마스크처럼 조용히 틀리는 걸 막는다.

사용: mem_equiv.py save <경로>     패치 전 기준값 저장
      mem_equiv.py check <경로>    패치 후 비교
"""
import os, sys, pathlib
import numpy as np

mode, out = sys.argv[1], pathlib.Path(sys.argv[2])

import jax, jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")
import openpi.models.model as _model
import openpi.training.config as _config
import etils.epath as epath

CFG = os.environ.get("MEM_CFG", "pi05_robosyn_items_handover_lora_cos82k")
CKPT = os.environ.get("MEM_CKPT", "/workspace/ckpt_dl/subtask/checkpoints/81999")

cfg = _config.get_config(CFG)
params = _model.restore_params(epath.Path(CKPT) / "params", dtype=jnp.float32)
model = cfg.model.load(params)
print("모델 로드", flush=True)

rng = np.random.default_rng(0)
img = jnp.asarray(rng.uniform(-1, 1, (1, 224, 224, 3)), dtype=jnp.float32)
tok, _ = model.PaliGemma.img(img, train=False)
tok = np.asarray(jax.device_get(tok), dtype=np.float64)
print(f"비전 토큰 {tok.shape}  합 {tok.sum():.8f}", flush=True)

if mode == "save":
    np.save(out, tok); print(f"저장 -> {out}")
else:
    ref = np.load(out)
    if ref.shape != tok.shape:
        sys.exit(f"✗ shape 불일치 {ref.shape} vs {tok.shape}")
    d = np.abs(ref - tok)
    print(f"최대 절대차 {d.max():.3e} · 평균 {d.mean():.3e}")
    print("K=1 동치 OK" if d.max() == 0 else
          ("비트 동일은 아니나 수치 오차 범위" if d.max() < 1e-6 else "✗ 다르다 — 구현 확인 필요"))
