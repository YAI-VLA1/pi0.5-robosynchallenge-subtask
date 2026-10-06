#!/usr/bin/env python3
"""롤아웃에서 **모델이 생성한** bbox 를 투영 GT 와 대조한다.

  python3 bbox_accuracy.py <롤아웃 데이터셋> <subtask 덤프>

왜 이 비교가 필요한가
  CE 가 내려간 것만으로는 grounding 을 증명하지 못한다. 이미지를 안 봐도
  팔 자세만으로 박스를 꽤 맞힐 수 있기 때문이다 (집기 전 구간 state-only 26.2 px).
  **그 선을 밑돌아야** 비전을 쓴 것이다.

GT 는 롤아웃 parquet 의 pen_pose 와 고정 카메라로 투영해 만든다 — 탐지기가 아니다.
박스를 못 낸 호출(<loc> 아닌 토큰)은 **버리지 않고** invalid 로 센다. 버리면
실패가 많을수록 성적이 좋아 보인다.
"""
from __future__ import annotations
import json, math, pathlib, sys
import numpy as np

sys.path.insert(0, "/root/rsc_recover")
W, H, PI0_STEP = 640, 480, 10
BASELINE_PX = 26.2        # items_handover 집기 전, state-only 선형회귀 실측


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def main():
    import pandas as pd
    from subtask_video import parse_dump
    from obb_labels import pen_corners, cam_high, project, min_area_rect, mat
    from make_bbox_labels import aabb_from_obb

    root = pathlib.Path(sys.argv[1])
    rows = parse_dump(sys.argv[2])
    K, T = cam_high(); C = pen_corners()
    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines()]

    calls = [math.ceil(e["length"] / PI0_STEP) for e in eps]
    if sum(calls) != len(rows):
        print(f"  (호출 수 {sum(calls)} vs 덤프 {len(rows)} — 완비된 앞부분만 쓴다)")
        acc = 0; keep = 0
        for c in calls:
            if acc + c > len(rows): break
            acc += c; keep += 1
        eps, calls = eps[:keep], calls[:keep]

    off = 0
    errs, ious, by_sub = [], [], {}
    n_gt = n_pred = 0
    for e, n in zip(eps, calls):
        ep = e["episode_index"]
        pen = mat(pd.read_parquet(root / f"data/chunk-000/episode_{ep:06d}.parquet",
                                  columns=["pen_pose"])["pen_pose"])
        for k in range(n):
            sent, box = rows[off + k]
            fr = min(k * PI0_STEP, len(pen) - 1)
            M = pen[fr]
            pw = (M[:3, :3] @ C.T).T + M[:3, 3]
            uv, z = project(K, T, pw)
            if (z <= 0).any():
                continue
            cx, cy, w_, h_, th = min_area_rect(uv)
            gx0, gy0, gx1, gy1 = aabb_from_obb(cx/W, cy/H, w_/W, h_/H, th)
            gt = [max(gx0, 0)*W, max(gy0, 0)*H, min(gx1, 1)*W, min(gy1, 1)*H]
            if gt[2] <= gt[0] or gt[3] <= gt[1]:
                continue
            n_gt += 1
            d = by_sub.setdefault(sent, [0, 0])
            d[1] += 1
            if box is None:
                continue
            n_pred += 1; d[0] += 1
            pr = [box[0]*W, box[1]*H, box[2]*W, box[3]*H]
            gc = ((gt[0]+gt[2])/2, (gt[1]+gt[3])/2)
            pc = ((pr[0]+pr[2])/2, (pr[1]+pr[3])/2)
            errs.append(math.hypot(pc[0]-gc[0], pc[1]-gc[1]))
            ious.append(iou(gt, pr))
        off += n

    e = np.array(errs); i = np.array(ious)
    print(f"GT 가 보이는 호출 {n_gt:,} · 모델이 박스를 낸 호출 {n_pred:,} "
          f"({n_pred/max(n_gt,1):.0%}) · invalid {n_gt-n_pred:,}")
    if len(e):
        print(f"  중심 오차   중앙값 {np.median(e):6.1f} px  ·  평균 {e.mean():6.1f}  ·  "
              f"90% {np.percentile(e,90):.1f}")
        print(f"  IoU        중앙값 {np.median(i):6.3f}  ·  >0.5 {np.mean(i>0.5):.0%} "
              f"(전체 GT 대비 {np.sum(i>0.5)/max(n_gt,1):.0%})")
        ok = np.median(e) < BASELINE_PX
        print(f"\n★ state-only 기준선 {BASELINE_PX} px 대비 "
              f"{'밑돎 — 비전을 쓰고 있다' if ok else '못 넘음 — 지름길 의심'}")
    print(f"\n{'subtask':46}{'박스 낸 비율':>12}{'호출':>8}")
    for s, (a, b) in sorted(by_sub.items(), key=lambda x: -x[1][1])[:10]:
        print(f"  {s[:44]:44}{a/max(b,1):11.0%}{b:8}")
    print("BBOX-ACCURACY-DONE")


if __name__ == "__main__":
    main()
