#!/usr/bin/env python3
"""학습 전 검증: subtask 가 실제로 프롬프트에 들어가는가.

파드 복구 때마다 여기서 깨졌다 (2026-09-26 에만 세 번). 학습을 띄우기 전에 반드시 돌릴 것.
확인 항목
  · token_loss_mask 가 존재하는가 (없으면 subtask 미주입)
  · 슬롯에 실제 문장 토큰이 있는가 (EOS 하나만 있으면 InjectSubtask 가 안 돈 것이다)
  · 프롬프트 형식이 `State: ...; Subtask: ...;\\nAction:` 인가 (Task 가 있으면 안 된다)
  · EOS 위치와 패딩 개수

import 순서는 scripts/train.py 를 그대로 따른다 (config 를 먼저 올리면 segfault).
/root 에 둔다 — /workspace 는 파드가 죽으면 사라진다.
"""
import dataclasses  # noqa: F401
import functools  # noqa: F401
import logging  # noqa: F401
import platform  # noqa: F401
from typing import Any  # noqa: F401

import etils.epath as epath  # noqa: F401
import flax.nnx as nnx  # noqa: F401
from flax.training import common_utils  # noqa: F401
import flax.traverse_util as traverse_util  # noqa: F401
import jax  # noqa: F401
import jax.experimental  # noqa: F401
import jax.numpy as jnp  # noqa: F401
import numpy as np
import optax  # noqa: F401
import tqdm_loggable.auto as tqdm  # noqa: F401
import wandb  # noqa: F401

import openpi.models.model as _model  # noqa: F401
import openpi.shared.array_typing as at  # noqa: F401
import openpi.shared.nnx_utils as nnx_utils  # noqa: F401
import openpi.training.checkpoints as _checkpoints  # noqa: F401
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.models.tokenizer as T


def main() -> None:
    c = _config.get_config("pi05_robosyn_items_handover_subtask_cos82k")
    dc = c.data.create(c.assets_dirs, c.model)
    print("repack:", [type(t).__name__ for t in dc.repack_transforms.inputs], flush=True)
    if "InjectSubtask" not in [type(t).__name__ for t in dc.repack_transforms.inputs]:
        raise SystemExit("FAIL: InjectSubtask 가 repack 그룹에 없다 — 엉뚱한 클래스에 배선됐다")

    dl = _data_loader.create_data_loader(c, num_batches=2, shuffle=False)
    obs, act = next(iter(dl))
    print("actions", act.shape, flush=True)
    if obs.token_loss_mask is None:
        raise SystemExit("FAIL: token_loss_mask 가 None — subtask 미주입")

    sp = T.PaligemmaTokenizer(c.model.max_token_len)._tokenizer      # noqa: SLF001
    bad = 0
    for b in range(min(3, obs.tokenized_prompt.shape[0])):
        toks = np.asarray(obs.tokenized_prompt[b])
        lm = np.asarray(obs.token_loss_mask[b])
        idx = np.flatnonzero(lm)
        slot = [int(x) for x in toks[idx]]
        eos = slot.index(1) if 1 in slot else -1
        sent = sp.decode(slot[:eos] if eos > 0 else [])
        full = sp.decode([int(x) for x in toks[np.asarray(obs.tokenized_prompt_mask[b])]])
        print(f"[{b}] 슬롯 {slot}", flush=True)
        print(f"    문장 {sent!r} · EOS {eos}/{len(slot)} · 학습자리 {eos + 1}", flush=True)
        print(f"    프롬프트 끝 {full[-90:]!r}", flush=True)
        if eos <= 0:
            print("    ★ 슬롯에 문장이 없다 (EOS 뿐) — InjectSubtask 가 안 돌았다", flush=True)
            bad += 1
        if "Task:" in full:
            print("    ★ 프롬프트에 Task 지시문이 남아 있다 — v2 패치 누락", flush=True)
            bad += 1
    if bad:
        raise SystemExit(f"FAIL: {bad}건")
    print("SMOKE-OK", flush=True)


if __name__ == "__main__":
    main()
