#!/usr/bin/env python3
"""MEM 켰을 때 학습 1스텝이 얼마나 느려지는지 잰다.

컴파일 1회 + 측정 N회. 실제 학습과 같은 trainable 필터를 쓴다.
PI05_MEM_FRAMES=1 / 6 으로 두 번 돌려 비교한다.
"""
import os, time
import etils.epath as epath
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
import optax
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main():
    T = int(os.environ.get("PI05_MEM_FRAMES", "1"))
    N = int(os.environ.get("MEM_STEPS", "10"))
    CFG = os.environ.get("MEM_CFG", "pi05_robosyn_items_handover_lora_cos82k")
    CKPT = os.environ.get("MEM_CKPT", "/workspace/ckpt_dl/subtask/checkpoints/81999")

    cfg = _config.get_config(CFG)
    import dataclasses as _dc
    _bs = int(os.environ.get("MEM_BATCH", "0"))
    if _bs:
        cfg = _dc.replace(cfg, batch_size=_bs)
    dl = _data_loader.create_data_loader(cfg, num_batches=1, shuffle=False)
    obs, act = next(iter(dl))
    print(f"T={T} batch={cfg.batch_size}", flush=True)

    model = cfg.model.load(_model.restore_params(epath.Path(CKPT) / "params", dtype=jnp.bfloat16))
    diff_state = nnx.DiffState(0, cfg.trainable_filter)
    tx = optax.adamw(1e-5)
    params = nnx.state(model, cfg.trainable_filter)
    opt_state = tx.init(params)

    @nnx.jit
    def step(model, opt_state, rng, obs, act):
        def f(m):
            return jnp.mean(m.compute_loss(rng, obs, act, train=True))
        loss, grads = nnx.value_and_grad(f, argnums=diff_state)(model)
        updates, opt_state = tx.update(grads, opt_state, nnx.state(model, cfg.trainable_filter))
        nnx.update(model, optax.apply_updates(nnx.state(model, cfg.trainable_filter), updates))
        return loss, opt_state

    rng = jax.random.key(0)
    t0 = time.time()
    loss, opt_state = step(model, opt_state, rng, obs, act)
    jax.block_until_ready(loss)
    print(f"컴파일+1스텝 {time.time()-t0:.1f}s · loss {float(loss):.4f}", flush=True)

    ts = []
    for i in range(N):
        t0 = time.time()
        loss, opt_state = step(model, opt_state, jax.random.key(i + 1), obs, act)
        jax.block_until_ready(loss)
        ts.append(time.time() - t0)
    ts = np.array(ts)
    print(f"스텝 시간 중앙값 {np.median(ts)*1000:.0f}ms · 평균 {ts.mean()*1000:.0f}ms "
          f"· min {ts.min()*1000:.0f}ms · max {ts.max()*1000:.0f}ms")
    print(f"82,000 스텝 환산 {np.median(ts)*82000/3600:.1f}시간")
    print(f"MEM-SPEED-DONE T={T} median_ms={np.median(ts)*1000:.1f}")


if __name__ == "__main__":
    main()
