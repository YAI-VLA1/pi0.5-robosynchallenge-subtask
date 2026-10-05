#!/usr/bin/env python3
"""bbox 예측을 채점한다 — **지름길을 걸러내는 게 목적**이다.

왜 필요한가 (실측)
  이미지를 안 봐도 박스를 꽤 맞힌다. 집기 전 구간에서
      상수(평균) 예측            61.8 px
      로봇 state 14축 선형회귀    26.2 px   <- 이미지 안 봄
  정책이 이미 펜 쪽으로 팔을 뻗고 있어서 팔 자세가 펜 위치를 알려준다. 순환이다.
  낮은 CE 가 grounding 을 증명하지 못한다.

그래서 세 가지를 같이 본다
  ① 예측 박스의 중심 오차 / IoU
  ② **state-only 기준선** 대비 (태스크마다 데이터에서 다시 계산하는 절차다)
  ③ **이미지 가림 ablation** — image_mask 를 끄고 오차가 거의 안 늘면 비전을 안 쓴 것

import 순서는 scripts/train.py 를 따른다.
"""
import dataclasses, os, sys
import etils.epath as epath
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

W, H, NBIN = 640, 480, 1024
LOC0 = 256000


def decode_box(ids):
    """슬롯 4토큰 -> (x0,y0,x1,y1) px. loc 토큰이 아니면 None."""
    v = [int(x) - LOC0 for x in ids]
    if any(t < 0 or t >= NBIN for t in v):
        return None
    ymin, xmin, ymax, xmax = [t / (NBIN - 1) for t in v]
    return xmin * W, ymin * H, xmax * W, ymax * H


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def main():
    sys.path.insert(0, "/root/rsc_recover")
    cfg_name = os.environ["BBOX_EVAL_CFG"]
    ckpt = os.environ["BBOX_EVAL_CKPT"]
    nb = int(os.environ.get("BBOX_EVAL_BATCHES", "20"))
    cfg = _config.get_config(cfg_name)
    cfg = dataclasses.replace(cfg, batch_size=int(os.environ.get("BBOX_EVAL_BATCH", "2")))

    dl = _data_loader.create_data_loader(cfg, num_batches=nb, shuffle=True)
    model = cfg.model.load(_model.restore_params(epath.Path(ckpt), dtype=jnp.bfloat16))

    @nnx.jit
    def gen(model, obs):
        return model.generate_subtask(obs) if hasattr(model, "generate_subtask") else obs

    res = {"full": [], "masked": []}
    for i, (obs, act) in enumerate(dl):
        lm = np.asarray(obs.token_loss_mask)
        tp = np.asarray(obs.tokenized_prompt)
        for tag in ("full", "masked"):
            o = obs
            if tag == "masked":
                o = dataclasses.replace(
                    obs, image_masks={k: jnp.zeros_like(v) for k, v in obs.image_masks.items()})
            out = gen(model, o)
            got = np.asarray(out.tokenized_prompt)
            for b in range(len(tp)):
                idx = np.flatnonzero(lm[b])
                if len(idx) < 4:
                    continue
                gt = decode_box(tp[b][idx[:4]])
                pr = decode_box(got[b][idx[:4]])
                if gt is None or pr is None:
                    continue
                gc = ((gt[0]+gt[2])/2, (gt[1]+gt[3])/2)
                pc = ((pr[0]+pr[2])/2, (pr[1]+pr[3])/2)
                res[tag].append((np.hypot(pc[0]-gc[0], pc[1]-gc[1]), iou(gt, pr)))

    print(f"{'':12}{'중심오차 중앙':>14}{'평균':>9}{'IoU 중앙':>10}{'IoU>0.5':>9}{'n':>7}")
    for tag in ("full", "masked"):
        a = np.array(res[tag])
        if not len(a):
            print(f"{tag:12}  (샘플 없음)"); continue
        print(f"{tag:12}{np.median(a[:,0]):11.1f} px{a[:,0].mean():9.1f}"
              f"{np.median(a[:,1]):10.3f}{np.mean(a[:,1]>0.5):9.1%}{len(a):7}")
    if len(res["full"]) and len(res["masked"]):
        f = np.median(np.array(res["full"])[:, 0]); m = np.median(np.array(res["masked"])[:, 0])
        print(f"\n이미지 가림 시 오차 증가 {m-f:+.1f} px ({m/max(f,1e-9):.2f}배)")
        print("  거의 안 늘면 비전을 안 쓰고 state 지름길만 탄 것이다.")
        print(f"\nstate-only 기준선 26.2 px (집기 전, items_handover 실측) 와 비교할 것.")
    print("BBOX-EVAL-DONE")


if __name__ == "__main__":
    main()
