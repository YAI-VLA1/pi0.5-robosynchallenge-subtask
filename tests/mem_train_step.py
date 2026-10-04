#!/usr/bin/env python3
"""MEM 켜고 학습 1스텝이 실제로 도는지 본다 (forward + backward).

shape 가 맞는 것과 학습이 도는 것은 다르다. 여기서 보는 것:
  · 데이터 배치가 모델에 들어가는가
  · loss 가 유한한가
  · 그래디언트가 흐르고 NaN 이 없는가
  · 비전 타워 파라미터 수가 그대로인가 (MEM 은 새 파라미터가 0 이어야 한다)

import 순서는 scripts/train.py 를 따른다 (config 를 먼저 올리면 segfault).
"""
import dataclasses, logging, os, platform  # noqa: F401
import etils.epath as epath
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
import optax  # noqa: F401
import openpi.models.model as _model
import openpi.shared.array_typing as at  # noqa: F401
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main():
    T = int(os.environ.get("PI05_MEM_FRAMES", "1"))
    CFG = os.environ.get("MEM_CFG", "pi05_robosyn_items_handover_lora_cos82k")
    CKPT = os.environ.get("MEM_CKPT", "/workspace/ckpt_dl/subtask/checkpoints/81999")

    cfg = _config.get_config(CFG)
    # 배치는 환경변수로 (T=6 · 배치 4 는 비전 역전파에만 16.4GiB 가 필요하다)
    import dataclasses as _dc
    _bs = int(os.environ.get("MEM_BATCH", "0"))
    if _bs:
        cfg = _dc.replace(cfg, batch_size=_bs)
    dl = _data_loader.create_data_loader(cfg, num_batches=1, shuffle=False)
    obs, act = next(iter(dl))
    print(f"배치: state {obs.state.shape} · actions {act.shape}", flush=True)
    for k, v in obs.images.items():
        print(f"  image {k:18} {v.shape}")
    if obs.frame_valid is not None:
        for k, v in obs.frame_valid.items():
            print(f"  frame_valid {k:12} {v.shape}  {np.asarray(v)[0].astype(int).tolist()}")
    else:
        print("  frame_valid 없음 (T=1)")

    model = cfg.model.load(_model.restore_params(epath.Path(CKPT) / "params", dtype=jnp.bfloat16))
    n_img = sum(int(np.prod(x.shape)) for k, x in
                jax.tree_util.tree_leaves_with_path(nnx.state(model, nnx.Param))
                if "img" in jax.tree_util.keystr(k))
    print(f"\n비전 타워 파라미터 {n_img:,}  (MEM 은 새 파라미터가 0 이어야 한다)", flush=True)

    # 실제 학습과 같은 trainable 필터를 쓴다. 안 그러면 동결된 LLM(2.9B)까지
    # 그래디언트를 잡아 메모리가 터진다 (MEM 때문이 아니다).
    diff_state = nnx.DiffState(0, cfg.trainable_filter)

    @nnx.jit
    def loss_and_grad(model, rng, obs, act):
        def f(m):
            return jnp.mean(m.compute_loss(rng, obs, act, train=True))
        return nnx.value_and_grad(f, argnums=diff_state)(model)

    loss, grads = loss_and_grad(model, jax.random.key(0), obs, act)
    print(f"\nloss {float(loss):.6f}  유한: {bool(jnp.isfinite(loss))}", flush=True)
    gl = [x for x in jax.tree_util.tree_leaves(grads) if hasattr(x, "shape")]
    gn = float(jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in gl)))
    nan = any(bool(jnp.any(~jnp.isfinite(g))) for g in gl)
    print(f"grad norm {gn:.6f}  NaN/Inf: {nan}")
    good = bool(jnp.isfinite(loss)) and not nan and gn > 0
    print("\nMEM-TRAIN-STEP " + ("OK" if good else "FAIL"))
    if not good:
        raise SystemExit(1)   # 자동 게이트용 종료 코드 (리뷰 지적)


if __name__ == "__main__":
    main()
