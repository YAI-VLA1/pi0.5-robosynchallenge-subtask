#!/usr/bin/env python3
"""체크포인트에서 flow 손실과 subtask CE 를 **따로** 잰다.

왜
  학습 로그의 loss 는 둘을 더한 값이라 가릴 수 없다. step 0 에서 7.29 가 나왔는데
  baseline 의 step 0 flow 가 0.136 이었으니 CE 가 약 52배로 지배한다.
  CE 는 빠르게 떨어질 것이나, 그 사이 action 학습이 밀리는지 확인해야 한다.

  학습을 건드리지 않으려고 오프라인으로 잰다 (train.py 에 has_aux 를 넣으면
  돌고 있는 잡을 재시작해야 하고, @at.typecheck 반환형도 건드려야 한다).

무엇을 보나
  · flow: baseline 같은 step 의 값과 비교 — 크게 나쁘면 CE 가중치를 낮춰야 한다
  · ce  : 내려가고 있는지 — 안 내려가면 subtask 를 못 배우는 것이다
  · subtask 정확도: teacher forcing 상태에서 argmax 가 GT 토큰과 맞는 비율

import 순서는 scripts/train.py 를 그대로 따른다 (config 를 먼저 올리면 segfault).
"""
import argparse
import dataclasses  # noqa: F401
import functools  # noqa: F401
import logging
import platform  # noqa: F401
from typing import Any  # noqa: F401

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils  # noqa: F401
import flax.traverse_util as traverse_util  # noqa: F401
import jax
import jax.experimental  # noqa: F401
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm  # noqa: F401

import openpi.models.model as _model
import openpi.shared.array_typing as at  # noqa: F401
import openpi.shared.nnx_utils as nnx_utils  # noqa: F401
import openpi.training.checkpoints as _checkpoints  # noqa: F401
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_robosyn_items_handover_subtask_cos82k")
    ap.add_argument("--ckpt", required=True, help="<...>/<step>/params 의 상위 <step> 디렉토리")
    ap.add_argument("--batches", type=int, default=8)
    a = ap.parse_args()

    cfg = _config.get_config(a.config)
    dl = _data_loader.create_data_loader(cfg, num_batches=a.batches, shuffle=True)

    model = cfg.model.load(_model.restore_params(epath.Path(a.ckpt) / "params", dtype=jnp.bfloat16))
    print(f"체크포인트 로드: {a.ckpt}", flush=True)

    @nnx.jit
    def parts(model, rng, obs, act):
        obs2 = _model.preprocess_observation(None, obs, train=False)
        # compute_loss 내부와 같은 계산을 복제하지 않고, CE 만 따로 부른다.
        # flow 는 CE 를 끈 상태의 compute_loss 로 얻는다.
        total = jnp.mean(model.compute_loss(rng, obs, act, train=False))
        return total, obs2

    rng = jax.random.key(0)
    tot, ces, accs, firsts = [], [], [], []
    for i, (obs, act) in enumerate(dl):
        rng, r = jax.random.split(rng)
        total, _ = parts(model, r, obs, act)
        tot.append(float(total))
        # CE 와 정확도는 prefix 를 한 번 더 돌려 직접 잰다
        # baseline config 에는 subtask 슬롯이 없다 (token_loss_mask 가 None).
        # 그 경우 total 이 곧 flow 다.
        if obs.token_loss_mask is None:
            ces.append(0.0); accs.append(float("nan")); firsts.append(float("nan"))
        else:
            ce, acc, acf = _ce_and_acc(model, obs)
            ces.append(ce); accs.append(acc); firsts.append(acf)
        if i + 1 >= a.batches:
            break

    t, c, ac = float(np.mean(tot)), float(np.mean(ces)), float(np.mean(accs))
    af = float(np.mean(firsts))
    print(f"\n배치 {len(tot)}개 평균")
    print(f"  total (flow + w*CE) : {t:.5f}")
    print(f"  subtask CE          : {c:.5f}")
    print(f"  flow (= total - CE) : {t - c:.5f}    <- baseline 같은 step 과 비교할 값")
    print(f"  subtask 토큰 정확도 : {ac:.4f}  (EOS 까지, teacher forcing)")
    print(f"  ★ 첫 토큰 정확도    : {af:.4f}  <- phase 를 장면에서 읽는가")


def _ce_and_acc(model, obs):
    """모델 안의 _rsc_subtask_ce 를 그대로 쓴다. 여기서 다시 구현하면 두 벌이 어긋난다."""
    o = _model.preprocess_observation(None, obs, train=False)
    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(o)
    from openpi.models.pi0 import make_attn_mask
    attn = make_attn_mask(prefix_mask, prefix_ar)
    pos = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), _ = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=pos)
    ce, acc, acc_first = model._rsc_subtask_ce(o, prefix_out)
    return float(jnp.mean(ce)), float(acc), float(acc_first)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
