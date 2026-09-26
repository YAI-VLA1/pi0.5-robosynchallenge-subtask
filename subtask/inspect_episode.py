#!/usr/bin/env python3
"""에피소드 하나를 훑으며 프레임마다 모델이 생성하는 subtask 문장을 본다.

왜
  validation set 을 따로 안 나눴다. 데이터셋이 1000 에피소드고 전부 학습에 들어갔다.
  다만 30k 스텝 x 배치 4 = 12만 샘플이고 전체 프레임이 32.7만 개라 개별 프레임의
  60% 이상은 아직 안 본 것이다 — 약한 의미의 held-out 이다. **train 데이터임을 잊지 말 것.**

  진짜 held-out 은 평가 롤아웃인데 거기엔 state 가 없다(영상만). 프롬프트에 state 가
  들어가야 해서 시뮬레이터 평가에서만 가능하다.

무엇을 보나
  프레임 -> 생성 문장 -> GT 문장. phase 를 따라가는지, 경계에서 흔들리는지,
  특히 7(인수 접근) <-> 8(그리퍼 닫기) 을 가르는지. 이 둘은 그리퍼 개도로만 갈려서
  step 1 의 분류기가 가장 약했던 지점이다.

import 순서는 train.py 를 따른다.
"""
import argparse
import dataclasses
import logging

import etils.epath as epath
import flax.nnx as nnx  # noqa: F401
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
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    cfg = _config.get_config(a.config)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    ds = _data_loader.transform_dataset(raw, dc)

    # 이 에피소드의 전역 인덱스 범위. items_handover 는 에피소드 길이가 327 로 고정이다
    # (2026-09-25 확인: 스케줄 복원값과 데이터셋 길이가 정확히 일치). lerobot 의
    # 메타데이터 API 는 버전마다 달라서 쓰지 않고, 대신 실제 값으로 검증한다.
    ep_len = int(raw[0]["frame_index"].shape[0]) if False else 327
    lo, hi = a.episode * ep_len, (a.episode + 1) * ep_len
    probe = raw[lo]
    got_ep = int(np.asarray(probe["episode_index"]).reshape(-1)[0])
    got_fr = int(np.asarray(probe["frame_index"]).reshape(-1)[0])
    if (got_ep, got_fr) != (a.episode, 0):
        raise SystemExit(
            f"인덱스 가정이 틀렸다: 전역 {lo} 이 에피소드 {got_ep} 프레임 {got_fr} 이다 "
            f"(에피소드 {a.episode} 프레임 0 을 기대했다). 에피소드 길이가 327 이 아닐 수 있다."
        )
    idxs = list(range(lo, hi, a.stride))
    print(f"에피소드 {a.episode}: 전역 인덱스 {lo}~{hi} ({hi - lo} 프레임), {len(idxs)}개 표본\n", flush=True)

    model = cfg.model.load(_model.restore_params(epath.Path(a.ckpt) / "params", dtype=jnp.bfloat16))
    sp = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)._tokenizer  # noqa: SLF001

    def cut(v):
        return v[: v.index(1)] if 1 in v else v

    n_ok = 0
    rows = []
    for s in range(0, len(idxs), a.batch):
        chunk = idxs[s : s + a.batch]
        items = [ds[i] for i in chunk]
        # image 는 카메라별 중첩 dict 다. 평탄하게 stack 하면 깨진다.
        batch = jax.tree.map(lambda *xs: np.stack(xs), *items)
        obs = _model.Observation.from_dict(batch)
        o = _model.preprocess_observation(None, obs, train=False)

        lm = np.asarray(o.token_loss_mask)
        gt = np.asarray(o.tokenized_prompt)
        blank = gt.copy()
        st = lm.argmax(axis=1)
        for b in range(len(chunk)):
            blank[b, int(st[b]) : int(st[b]) + 14] = 0
        gen = np.asarray(
            model._rsc_generate_subtask(                                # noqa: SLF001
                dataclasses.replace(o, tokenized_prompt=jnp.asarray(blank))
            ).tokenized_prompt
        )

        for b, gidx in enumerate(chunk):
            p = int(st[b])
            g = sp.decode(cut([int(x) for x in gen[b, p : p + 14]])).strip()
            t = sp.decode(cut([int(x) for x in gt[b, p : p + 14]])).strip()
            frame = gidx - lo
            ok = g == t
            n_ok += ok
            rows.append((frame, ok, g, t))
            print(f"  f{frame:>4}  {'O' if ok else 'X'}  {g!r}"
                  + ("" if ok else f"\n            GT: {t!r}"), flush=True)

    print(f"\n일치 {n_ok}/{len(rows)} ({n_ok / max(len(rows),1):.1%})  — train 데이터임에 유의")
    bad = [r for r in rows if not r[1]]
    if bad:
        print("\n틀린 프레임")
        for f, _, g, t in bad:
            print(f"  f{f:>4}  생성 {g!r}\n         정답 {t!r}")
    print("INSPECT-DONE", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
