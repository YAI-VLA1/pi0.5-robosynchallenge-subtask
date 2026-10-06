#!/usr/bin/env python3
"""실제 배치로 '박스 없음' CE 와 집기 가중을 확인한다.

보는 것
  ① visible=2(화면 밖) 프레임이 <loc0000> 네 개로 **감독되는가** (pad 아님)
  ② 집기 구간 프레임의 loss_weight 가 올라가는가
  ③ 그 밖 프레임은 1.0 인가
"""
import dataclasses, os, sys
import etils.epath as epath  # noqa: F401
import flax.nnx as nnx  # noqa: F401
import jax  # noqa: F401
import numpy as np
import openpi.models.model as _model  # noqa: F401
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

CFG = os.environ.get("SMOKE_CFG", "pi05_robosyn_items_handover_subtask_cos82k")


def main():
    sys.path.insert(0, "/root/rsc_recover")
    cfg = _config.get_config(CFG)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    tok = next(t.tokenizer for t in dc.model_transforms.inputs
               if isinstance(t, _transforms.TokenizePrompt))
    sp = tok._tokenizer
    chain = [*dc.repack_transforms.inputs, *dc.data_transforms.inputs,
             *dc.model_transforms.inputs]

    import glob
    # visible=2 인 프레임과 집기 구간 프레임을 찾는다
    lens = [len(np.load(f)) for f in sorted(glob.glob("/workspace/bbox_tokens/*.npy"))]
    base = {}
    acc = 0
    for i, n in enumerate(lens):
        base[i] = acc; acc += n
    tgt = {}
    for ep in range(30):
        t = np.load(f"/workspace/bbox_tokens/ep{ep:04d}.npy")
        z = np.flatnonzero(t[:, 4] == 2)
        if len(z) and "none" not in tgt:
            tgt["none"] = (ep, int(z[0]))
        if "grasp" not in tgt:
            tgt["grasp"] = (ep, 80)
        if "other" not in tgt:
            tgt["other"] = (ep, 10)
        if len(tgt) == 3: break

    ok = {}
    for tag, (ep, fr) in tgt.items():
        d = raw[base[ep] + fr]
        for t in chain:
            d = t(d)
        ids = np.asarray(d["tokenized_prompt"]); m = np.asarray(d["tokenized_prompt_mask"])
        lm = np.asarray(d["token_loss_mask"]).astype(bool)
        cov = [int(x) for x in ids[lm]]
        nloc = sum(1 for x in cov[:4] if 256000 <= x <= 257023)
        w = float(d.get("loss_weight", 1.0))
        print(f"[{tag:6}] ep{ep} f{fr}  감독 {len(cov)}칸 · bbox 자리 <loc> {nloc}/4 · "
              f"loss_weight {w}")
        print(f"         bbox 토큰 {cov[:4]}")
        if tag == "none":
            ok["① '박스 없음' 이 <loc0000>×4 로 감독됨"] = (nloc == 4 and all(x == 256000 for x in cov[:4]))
        if tag == "grasp":
            ok["② 집기 구간 가중 > 1"] = (w > 1.0)
        if tag == "other":
            ok["③ 그 밖은 1.0"] = (w == 1.0)

    print()
    for k, v in ok.items():
        print(f"  {'OK ' if v else '✗  '} {k}")
    print("\nGRASP-SMOKE " + ("OK" if all(ok.values()) else "FAIL"))
    if not all(ok.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
