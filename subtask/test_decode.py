#!/usr/bin/env python3
"""추론 디코드 실검증 — 모델이 실제로 subtask 문장을 생성하는가.

학습은 teacher forcing 이라 GT 문장이 프롬프트에 들어 있다. 추론에는 없다.
`_rsc_generate_subtask` 가 빈 슬롯을 채우는 경로는 지금까지 한 번도 실행된 적이 없다.

무엇을 보나
  · 생성된 문장이 문법적으로 말이 되는가 (아무 토큰이나 뱉지 않는가)
  · GT 와 얼마나 일치하는가 (프레임별 phase 를 맞히는가)
  · EOS 가 제대로 찍히는가 (슬롯 끝까지 쓰레기를 채우지 않는가)

import 순서는 train.py 를 따른다.
"""
import argparse
import dataclasses
import logging

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils  # noqa: F401
import flax.traverse_util as traverse_util  # noqa: F401
import jax
import jax.experimental  # noqa: F401
import jax.numpy as jnp
import numpy as np
import optax  # noqa: F401
import tqdm_loggable.auto as tqdm  # noqa: F401

import openpi.models.model as _model
import openpi.shared.array_typing as at  # noqa: F401
import openpi.shared.nnx_utils as nnx_utils  # noqa: F401
import openpi.training.checkpoints as _checkpoints  # noqa: F401
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.models.tokenizer as _tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_robosyn_items_handover_subtask_cos82k")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=3)
    a = ap.parse_args()

    cfg = _config.get_config(a.config)
    dl = _data_loader.create_data_loader(cfg, num_batches=a.batches, shuffle=True)
    model = cfg.model.load(_model.restore_params(epath.Path(a.ckpt) / "params", dtype=jnp.bfloat16))
    tk = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)
    sp = tk._tokenizer
    print(f"체크포인트: {a.ckpt}\n", flush=True)

    n_ok = n_tot = 0
    for bi, (obs, _act) in enumerate(dl):
        o = _model.preprocess_observation(None, obs, train=False)
        lm = np.asarray(o.token_loss_mask)
        gt_tok = np.asarray(o.tokenized_prompt)

        # 슬롯을 비우고(패딩) 모델이 채우게 한다 — 추론과 같은 조건
        blank = gt_tok.copy()
        starts = lm.argmax(axis=1)
        for b in range(len(blank)):
            s = int(starts[b])
            blank[b, s:s + 14] = 0
        o_blank = dataclasses.replace(o, tokenized_prompt=jnp.asarray(blank))

        gen = model._rsc_generate_subtask(o_blank)
        gen_tok = np.asarray(gen.tokenized_prompt)

        for b in range(len(blank)):
            s = int(starts[b])
            g = [int(x) for x in gen_tok[b, s:s + 14]]
            t = [int(x) for x in gt_tok[b, s:s + 14]]
            cut = lambda v: v[: v.index(1)] if 1 in v else v          # EOS(1) 앞까지
            gs, ts = sp.decode(cut(g)), sp.decode(cut(t))
            ok = gs.strip() == ts.strip()
            n_ok += ok
            n_tot += 1
            mark = "O" if ok else "X"
            print(f"[{bi}.{b}] {mark}", flush=True)
            print(f"      생성: {gs!r}", flush=True)
            if not ok:
                print(f"      정답: {ts!r}", flush=True)
            eos = g.index(1) if 1 in g else -1
            print(f"      EOS 위치 {eos} / 14, 원시토큰 {g}", flush=True)

    print(f"\n문장 일치 {n_ok}/{n_tot} ({n_ok / max(n_tot,1):.1%})", flush=True)
    print("DECODE-TEST-DONE", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
