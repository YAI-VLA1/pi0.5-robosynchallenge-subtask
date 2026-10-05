#!/usr/bin/env python3
"""체크포인트가 **어떤 설정으로 학습됐는지**를 loss 로 역추적한다.

왜 되나
  프롬프트 형식이 학습과 다르면 모델은 조용히 망가지지 않고 **loss 가 폭발**한다
  (2026-09-26: 형식이 달라져 재개 loss 가 0.015 -> 10.25 로 튀었다).
  그래서 설정을 바꿔 가며 forward 만 돌려 보면 맞는 조합이 드러난다.

  기울기·optimizer 를 안 만들므로 24GB 에서도 돈다 (학습 1스텝은 OOM 이다).

사용
  PI05_SUBTASK_W=... PI05_MISTAKE=... PI05_MEM_FRAMES=... PROBE_CFG=<config> \
  PROBE_CKPT=<.../params> python3 probe_ckpt_config.py

import 순서는 scripts/train.py 를 따른다 (config 먼저 올리면 segfault).
"""
import os
import etils.epath as epath
import flax.nnx as nnx  # noqa: F401
import jax, jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.shared.array_typing as at  # noqa: F401
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main():
    cfg_name = os.environ["PROBE_CFG"]
    ckpt = os.environ["PROBE_CKPT"]
    nb = int(os.environ.get("PROBE_BATCHES", "4"))
    cfg = _config.get_config(cfg_name)
    import dataclasses as _dc
    bs = int(os.environ.get("PROBE_BATCH", "2"))
    cfg = _dc.replace(cfg, batch_size=bs)

    dl = _data_loader.create_data_loader(cfg, num_batches=nb, shuffle=True)
    model = cfg.model.load(_model.restore_params(epath.Path(ckpt), dtype=jnp.bfloat16))

    @nnx.jit
    def loss_fn(model, rng, obs, act):
        return jnp.mean(model.compute_loss(rng, obs, act, train=False))

    losses = []
    for i, (obs, act) in enumerate(dl):
        l = float(loss_fn(model, jax.random.key(i), obs, act))
        losses.append(l)
    a = np.array(losses)
    tag = (f"SUBTASK_W={os.environ.get('PI05_SUBTASK_W','0')} "
           f"MISTAKE={os.environ.get('PI05_MISTAKE','0')} "
           f"MEM={os.environ.get('PI05_MEM_FRAMES','1')}")
    print(f"PROBE {cfg_name} | {tag} | loss 중앙값 {np.median(a):.5f} "
          f"평균 {a.mean():.5f} (배치 {len(a)}개: {np.round(a,4).tolist()})")


if __name__ == "__main__":
    main()
