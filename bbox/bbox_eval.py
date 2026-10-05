#!/usr/bin/env python3
"""bbox 예측을 채점한다 — **지름길을 걸러내는 게 목적**이다.

왜 필요한가 (실측)
  이미지를 안 봐도 박스를 꽤 맞힌다. 집기 전 구간에서
      상수(평균) 예측            61.8 px
      로봇 state 14축 선형회귀    26.2 px   <- 이미지 안 봄
  정책이 이미 펜 쪽으로 팔을 뻗고 있어서 팔 자세가 펜 위치를 알려준다. 순환이다.
  낮은 CE 가 grounding 을 증명하지 못한다.

세 가지를 같이 본다
  ① 예측 박스의 중심 오차 / IoU
  ② state-only 기준선 대비 (태스크마다 데이터에서 다시 계산하는 **절차**다)
  ③ **이미지 가림 ablation** — 오차가 거의 안 늘면 비전을 안 쓴 것

★ 평가 자체의 함정 두 개를 막는다 (외부 리뷰)
  · 생성 메서드 이름이 틀리면 조용히 GT 를 그대로 채점해 **오차 0 / IoU 1** 이 나온다.
    메서드가 없으면 멈춘다.
  · 생성 전에 GT 슬롯을 **반드시 비운다**. 안 비우면 모델이 정답을 보고 베낀다.
  · loc 이외 토큰을 낸 샘플을 버리면 실패가 많을수록 성적이 좋아 보인다.
    **전체 GT 샘플을 분모**로 두고 invalid 비율을 따로 보고한다.
"""
import dataclasses, os, sys
import etils.epath as epath
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

W, H, NBIN, LOC0, PAD = 640, 480, 1024, 256000, 0
BBOX_SLOT = 4


def decode_box(ids):
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
    cfg = _config.get_config(os.environ["BBOX_EVAL_CFG"])
    cfg = dataclasses.replace(cfg, batch_size=int(os.environ.get("BBOX_EVAL_BATCH", "2")))
    nb = int(os.environ.get("BBOX_EVAL_BATCHES", "20"))
    dl = _data_loader.create_data_loader(cfg, num_batches=nb, shuffle=True)
    model = cfg.model.load(_model.restore_params(
        epath.Path(os.environ["BBOX_EVAL_CKPT"]), dtype=jnp.bfloat16))

    if not hasattr(model, "_rsc_generate_subtask"):
        sys.exit("★ 모델에 _rsc_generate_subtask 가 없다 — 디코드 패치가 안 들어갔다. "
                 "이름이 틀리면 조용히 GT 를 채점해 오차 0 이 나온다.")

    @nnx.jit
    def gen(model, obs):
        return model._rsc_generate_subtask(obs)

    def blank(obs):
        """{MARK}: 생성 전에 두 슬롯을 pad 로 비운다. 안 비우면 모델이 정답을 베낀다."""
        tok = obs.tokenized_prompt
        lm = obs.token_loss_mask.astype(bool)
        return dataclasses.replace(obs, tokenized_prompt=jnp.where(lm, PAD, tok))

    res = {"full": [], "masked": []}
    n_gt = n_invalid = {"full": 0, "masked": 0}
    n_gt = {"full": 0, "masked": 0}; n_invalid = {"full": 0, "masked": 0}
    for i, (obs, act) in enumerate(dl):
        lm = np.asarray(obs.token_loss_mask)
        tp = np.asarray(obs.tokenized_prompt)
        for tag in ("full", "masked"):
            o = blank(obs)
            if tag == "masked":
                o = dataclasses.replace(
                    o, image_masks={k: jnp.zeros_like(v) for k, v in o.image_masks.items()})
            got = np.asarray(gen(model, o).tokenized_prompt)
            for b in range(len(tp)):
                idx = np.flatnonzero(lm[b])[:BBOX_SLOT]
                if len(idx) < BBOX_SLOT:
                    continue
                gt = decode_box(tp[b][idx])
                if gt is None:          # 라벨이 없는 프레임 — 분모에서 뺀다
                    continue
                n_gt[tag] += 1
                pr = decode_box(got[b][idx])
                if pr is None:          # loc 이 아닌 토큰을 냈다 — 실패로 센다
                    n_invalid[tag] += 1
                    continue
                gc = ((gt[0]+gt[2])/2, (gt[1]+gt[3])/2)
                pc = ((pr[0]+pr[2])/2, (pr[1]+pr[3])/2)
                res[tag].append((np.hypot(pc[0]-gc[0], pc[1]-gc[1]), iou(gt, pr)))

    print(f"{'':10}{'GT 샘플':>9}{'invalid':>9}{'중심오차 중앙':>14}{'IoU 중앙':>10}"
          f"{'IoU>0.5 (전체 대비)':>20}")
    med = {}
    for tag in ("full", "masked"):
        a = np.array(res[tag]); n = n_gt[tag]
        if not n:
            print(f"{tag:10}  (샘플 없음)"); continue
        hit = float(np.sum(a[:, 1] > 0.5)) if len(a) else 0.0
        med[tag] = np.median(a[:, 0]) if len(a) else float("nan")
        print(f"{tag:10}{n:9}{n_invalid[tag]:9}"
              f"{med[tag]:11.1f} px{(np.median(a[:,1]) if len(a) else float('nan')):10.3f}"
              f"{hit/n:20.1%}")
    if len(med) == 2:
        print(f"\n이미지 가림 시 오차 증가 {med['masked']-med['full']:+.1f} px "
              f"({med['masked']/max(med['full'],1e-9):.2f}배)")
        print("  거의 안 늘면 비전을 안 쓰고 state 지름길만 탄 것이다.")
    print("\nstate-only 기준선 26.2 px (items_handover 집기 전 실측) 를 확실히 밑돌아야 한다.")
    print("BBOX-EVAL-DONE")


if __name__ == "__main__":
    main()
