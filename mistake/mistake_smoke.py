#!/usr/bin/env python3
"""mistake 배선 검증: 토크나이즈된 프롬프트를 되돌려 눈으로 확인한다.

shape 가 맞는 것과 프롬프트가 맞는 것은 다르다. 2026-09-26 에 프롬프트 형식이
달라져 재개 loss 가 0.015 -> 10.25 로 튄 적이 있다. 그래서 문자열을 직접 본다.

확인 항목
  ① 데모 프레임            Mistake: false
  ② 롤아웃 t_mistake 이전   Mistake: false
  ③ 롤아웃 t_mistake 이후   Mistake: true
  ④ 드롭(dropout)          필드가 통째로 빠지고 **기존 형식과 완전히 같다**
  ⑤ 추론 경로(repack 없음)  Mistake: false
  ⑥ 토큰 길이가 max_token_len 을 넘지 않는다

import 순서는 scripts/train.py 를 따른다 (config 를 먼저 올리면 segfault).
"""
import dataclasses, json, os, pathlib, sys  # noqa: F401
import etils.epath as epath  # noqa: F401
import flax.nnx as nnx  # noqa: F401
import jax  # noqa: F401
import numpy as np
import optax  # noqa: F401
import openpi.models.model as _model  # noqa: F401
import openpi.shared.array_typing as at  # noqa: F401
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


CFG = os.environ.get("MISTAKE_CFG", "pi05_robosyn_items_handover_mistake_cos82k")
_PI05 = pathlib.Path(os.environ.get(
    "PI05", "/workspace/rsc_ws/RoboSynChallenge/policy/pi05"))
DS  = _PI05 / "training_data/RoboSynChallenge/cobotmagic_Sim_items_handover_mix"


def main():
    sys.path.insert(0, "/root/rsc_recover")
    import lerobot_offline; lerobot_offline.patch()

    cfg = _config.get_config(CFG)
    dc  = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    tok = None
    for t in dc.model_transforms.inputs:
        if isinstance(t, _transforms.TokenizePrompt):
            tok = t.tokenizer
    assert tok is not None, "TokenizePrompt 를 못 찾았다"
    sp = tok._tokenizer
    maxlen = cfg.model.max_token_len

    starts = {int(k): int(v) for k, v in
              json.loads((DS / "meta/mistake_starts.json").read_text()).items()}
    lengths = {json.loads(l)["episode_index"]: json.loads(l)["length"]
               for l in (DS / "meta/episodes.jsonl").read_text().splitlines()}
    frm = {}
    acc = 0
    for e in sorted(lengths):
        frm[e] = acc
        acc += lengths[e]

    def decode(global_idx, dropout):
        """repack -> data -> model 변환을 직접 돌려 프롬프트 문자열을 되돌린다."""
        chain = [*dc.repack_transforms.inputs, *dc.data_transforms.inputs,
                 *dc.model_transforms.inputs]
        chain = [dataclasses.replace(t, dropout=dropout)
                 if isinstance(t, _transforms.InjectMistake) else t for t in chain]
        d = raw[global_idx]
        for t in chain:
            d = t(d)
        ids = np.asarray(d["tokenized_prompt"])
        m = np.asarray(d["tokenized_prompt_mask"])
        return sp.decode([int(x) for x in ids[m]]), int(m.sum())

    ep_mis = sorted(starts)[0]
    t_mis  = starts[ep_mis]
    cases = [
        ("① 데모 ep0 f0",               frm[0] + 0,                  0.0),
        ("② 롤아웃 mistake 이전",        frm[ep_mis] + max(0, t_mis - 5), 0.0),
        ("③ 롤아웃 mistake 이후",        frm[ep_mis] + t_mis + 5,     0.0),
        ("④ 드롭 (dropout=1.0)",        frm[ep_mis] + t_mis + 5,     1.0),
    ]
    out, lens = {}, {}
    for name, idx, dp in cases:
        txt, n = decode(idx, dp)
        out[name], lens[name] = txt, n
        print(f"{name}\n    [{n:3}토큰] {txt}\n")

    # ⑤ 추론 경로: repack 이 안 돌아 mistake 키가 없다
    chain = [*dc.data_transforms.inputs, *dc.model_transforms.inputs]
    d = dict(raw[frm[0]])
    d = _transforms.RepackTransform({
        "observation/image": "observation.images.cam_high",
        "observation/left_wrist_image": "observation.images.cam_left_wrist",
        "observation/right_wrist_image": "observation.images.cam_right_wrist",
        "observation/state": "observation.state",
        "prompt": "prompt"})(d)
    for t in chain:
        d = t(d)
    ids = np.asarray(d["tokenized_prompt"]); m = np.asarray(d["tokenized_prompt_mask"])
    infer = sp.decode([int(x) for x in ids[m]])
    print(f"⑤ 추론 경로 (repack 없음)\n    [{int(m.sum()):3}토큰] {infer}\n")

    ok = {
        "① 데모 = Mistake: false":        "Mistake: false" in out["① 데모 ep0 f0"],
        "② 이전 = Mistake: false":        "Mistake: false" in out["② 롤아웃 mistake 이전"],
        "③ 이후 = Mistake: true":         "Mistake: true"  in out["③ 롤아웃 mistake 이후"],
        "④ 드롭 = 필드 없음":              "Mistake" not in out["④ 드롭 (dropout=1.0)"],
        "⑤ 추론 = Mistake: false":        "Mistake: false" in infer,
        "⑥ 토큰 길이 < max":              max(lens.values()) < maxlen,
    }
    print(f"max_token_len {maxlen} · 최장 프롬프트 {max(lens.values())} 토큰\n")
    for k, v in ok.items():
        print(f"  {'OK ' if v else '✗  '} {k}")
    print("\nMISTAKE-SMOKE " + ("OK" if all(ok.values()) else "FAIL"))


if __name__ == "__main__":
    main()
