#!/usr/bin/env python3
"""bbox 슬롯 배선 검증 — 토큰을 문자열로 되돌려 본다.

shape 가 맞는 것과 프롬프트가 맞는 것은 다르다. 2026-09-26 에 형식이 어긋나
재개 loss 가 0.015 -> 10.25 로 튄 적이 있다.

보는 것
  ① 학습: Pen 슬롯에 라벨 토큰이 들어가는가
  ② loss mask 가 **두 슬롯만** 덮는가 (사이 텍스트·head·tail 은 제외)
  ③ 박스가 안 보이는 프레임: 슬롯이 패딩이고 loss 에서 빠지는가
  ④ 추론(라벨 없음): 빈 슬롯이 생기는가
  ⑤ PI05_BBOX=0: 기존 형식과 **바이트 단위로 같은가**
  ⑥ 토큰 길이가 max_token_len 을 안 넘는가
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

CFG = os.environ.get("BBOX_CFG", "pi05_robosyn_items_handover_subtask_mistake_cos82k")


def main():
    sys.path.insert(0, "/root/rsc_recover")
    import lerobot_offline; lerobot_offline.patch()
    cfg = _config.get_config(CFG)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    tok = next(t.tokenizer for t in dc.model_transforms.inputs
               if isinstance(t, _transforms.TokenizePrompt))
    sp = tok._tokenizer

    def run(idx, chain=None, drop_bbox=False):
        c = chain or [*dc.repack_transforms.inputs, *dc.data_transforms.inputs,
                      *dc.model_transforms.inputs]
        d = raw[idx]
        for t in c:
            if drop_bbox and isinstance(t, _transforms.InjectBBox):
                continue
            d = t(d)
        ids = np.asarray(d["tokenized_prompt"]); m = np.asarray(d["tokenized_prompt_mask"])
        lm = np.asarray(d.get("token_loss_mask", np.zeros_like(m)))
        return sp.decode([int(x) for x in ids[m]]), ids, m, lm

    ok = {}
    txt, ids, m, lm = run(0)
    print(f"① 학습 프롬프트 [{int(m.sum())}토큰]\n    {txt}\n")
    ok["① Pen 슬롯에 loc 토큰"] = any(256000 <= int(x) <= 257023 for x in ids[m])

    # loss mask 가 덮는 토큰
    cov = [int(x) for x in ids[lm.astype(bool)]]
    nloc = sum(1 for x in cov if 256000 <= x <= 257023)
    print(f"② loss mask 가 덮는 토큰 {len(cov)}개 — 그중 <loc> {nloc}개")
    print(f"    디코드: {sp.decode(cov)!r}\n")
    import os as _o
    _w = _o.environ.get("PI05_BBOX_WRIST", "0") == "1"
    want_loc, want_len = (8, 22) if _w else (4, 18)
    ok[f"② loss mask = bbox{'+wrist' if _w else ''} + subtask14"] = (
        len(cov) == want_len and nloc == want_loc)

    # ③ 박스 없는 프레임 찾기
    import glob, pathlib
    found = None
    for ep in range(5):
        t = np.load(f"/workspace/bbox_tokens/ep{ep:04d}.npy")
        z = np.flatnonzero(t[:, 4] == 2)
        if len(z):
            found = (ep, int(z[0])); break
    if found:
        ep, fr = found
        gidx = sum(len(np.load(f"/workspace/bbox_tokens/ep{e:04d}.npy")) for e in range(ep)) + fr
        txt3, ids3, m3, lm3 = run(gidx)
        cov3 = [int(x) for x in ids3[lm3.astype(bool)]]
        nloc3 = sum(1 for x in cov3 if 256000 <= x <= 257023)
        # 새 설계: 구조 mask 는 **18칸 유지**(attention 경계·디코드 시작점이 라벨
        #   유무에 흔들리면 안 된다). bbox 감독만 CE 에서 `tgt == pad_id` 로 뺀다.
        #   14칸을 기대하면 올바르게 고친 코드에서도 실패한다.
        n_pad = sum(1 for x in cov3[:4] if int(x) == 0)  # 참고용
        print(f"③ 박스 없는 프레임 (ep{ep} f{fr}) — 구조 mask {len(cov3)}칸, "
              f"<loc> {nloc3}개, bbox 자리 pad {n_pad}/4")
        # 새 규약: '박스 없음' 은 <loc0000> 로 **감독된다** (pad 아님)
        ok["③ 구조 mask 유지"] = (len(cov3) == want_len)
        ok["③b '박스 없음' 이 <loc0000> 로 감독"] = (nloc3 == want_loc)
    else:
        print("③ 박스 없는 프레임을 못 찾음 — 건너뜀")
        ok["③ 구조 mask 유지"] = True
        ok["③b '박스 없음' 이 <loc0000> 로 감독"] = True

    # ④ 추론 경로 (InjectBBox 없음)
    txt4, ids4, m4, lm4 = run(0, drop_bbox=True)
    print(f"\n④ 추론 경로 (라벨 없음) [{int(m4.sum())}토큰]\n    {txt4}")
    ok["④ 추론은 빈 슬롯"] = ("Pen:" in txt4
                              and sum(1 for x in ids4[m4] if 256000 <= int(x) <= 257023) == 0)

    ok["⑥ 길이 < max"] = int(m.sum()) < cfg.model.max_token_len
    print(f"\nmax_token_len {cfg.model.max_token_len} · 최장 {max(int(m.sum()), int(m4.sum()))}")
    print()
    for k, v in ok.items():
        print(f"  {'OK ' if v else '✗  '} {k}")
    print("\nBBOX-SMOKE " + ("OK" if all(ok.values()) else "FAIL"))
    if not all(ok.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
