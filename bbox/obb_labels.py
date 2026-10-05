#!/usr/bin/env python3
"""펜의 **회전 2D 박스**를 학습 데이터 전체에 뽑는다.

왜 투영인가
  학습 parquet 에 pen_pose (4x4 월드 변환) 가 들어 있고, cam_high 는
  **고정 카메라**라 intrinsics/extrinsics 가 gym_config 에 박혀 있다.
  그래서 시뮬레이터도 탐지기도 없이 **정확한 GT** 를 계산할 수 있다.
  (YOLO 축정렬 박스는 펜이 비스듬하면 거의 정사각형이 되어 — 실측 32% —
   그리퍼를 어느 각도로 내릴지 알려주지 못한다.)

산출  (cx, cy, w, h, theta)
  cx,cy  박스 중심 (0~1 정규화)
  w,h    긴 변 / 짧은 변 (0~1 정규화)
  theta  긴 변의 방향, 0~pi (펜은 앞뒤 구분이 무의미하므로 180도면 충분)

검증  투영 박스의 축정렬 외접 사각형 vs YOLO 박스 IoU 를 같이 낸다.
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import numpy as np

CFG = ("/workspace/rsc_ws/RoboSynChallenge/configs/items_handover/random/gym_config.json")
OBJ = "/workspace/rsc_ws/RoboSynChallenge/assets/Pen/the_pen.obj"
DS  = ("/workspace/rsc_ws/RoboSynChallenge/policy/pi05/training_data"
       "/RoboSynChallenge/cobotmagic_Sim_items_handover")
W, H = 640, 480


def pen_corners():
    """펜 로컬 OBB 의 8 꼭짓점 (body_scale 적용)."""
    g = json.load(open(CFG))
    ro = [o for o in g.get("rigid_object", []) if o.get("uid") == "pen"][0]
    sc = np.array(ro.get("body_scale", [1, 1, 1]), float)
    v = np.array([[float(x) for x in l.split()[1:4]]
                  for l in pathlib.Path(OBJ).read_text().splitlines()
                  if l.startswith("v ")])
    lo, hi = v.min(0) * sc, v.max(0) * sc
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                               for y in (lo[1], hi[1])
                               for z in (lo[2], hi[2])], float)


def cam_high():
    """(K, world->camera 4x4). OpenCV 규약: x 오른쪽, y 아래, z 앞."""
    g = json.load(open(CFG))
    c = [s for s in g["sensor"] if s.get("uid") == "cam_high"][0]
    fx, fy, cx, cy = c["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    e = c["extrinsics"]
    eye = np.array(e["eye"], float); tgt = np.array(e["target"], float)
    up = np.array(e["up"], float)
    f = tgt - eye; f /= np.linalg.norm(f)          # z (앞)
    r = np.cross(f, up); r /= np.linalg.norm(r)    # x (오른쪽)
    d = np.cross(f, r)                             # y (아래)  = f × x
    Rwc = np.stack([r, d, f])                      # 행이 카메라 축
    T = np.eye(4); T[:3, :3] = Rwc; T[:3, 3] = -Rwc @ eye
    return K, T


def project(K, T, pts_w):
    p = (T @ np.c_[pts_w, np.ones(len(pts_w))].T).T[:, :3]
    z = np.maximum(p[:, 2], 1e-6)
    uv = (K @ (p / z[:, None]).T).T[:, :2]
    return uv, p[:, 2]


def min_area_rect(pts):
    """회전 캘리퍼스. cv2 없이 — 볼록껍질 변마다 축정렬 넓이를 재 최소를 고른다."""
    P = np.asarray(pts, float)
    # 볼록껍질 (단조 체인)
    s = P[np.lexsort((P[:, 1], P[:, 0]))]
    def half(q):
        h = []
        for p in q:
            while len(h) >= 2 and np.cross(h[-1]-h[-2], p-h[-2]) <= 0: h.pop()
            h.append(p)
        return h
    hull = np.array(half(s)[:-1] + half(s[::-1])[:-1])
    if len(hull) < 3:
        lo, hi = P.min(0), P.max(0)
        return (*(lo+hi)/2, *(hi-lo), 0.0)
    best = None
    for i in range(len(hull)):
        e = hull[(i+1) % len(hull)] - hull[i]
        n = np.linalg.norm(e)
        if n < 1e-9: continue
        u = e / n; v = np.array([-u[1], u[0]])
        a = hull @ u; b = hull @ v
        wdt, hgt = a.max()-a.min(), b.max()-b.min()
        if best is None or wdt*hgt < best[0]:
            ctr = (a.min()+a.max())/2*u + (b.min()+b.max())/2*v
            best = (wdt*hgt, ctr, wdt, hgt, math.atan2(u[1], u[0]))
    _, ctr, wdt, hgt, th = best
    if hgt > wdt:                       # 긴 변을 w 로
        wdt, hgt = hgt, wdt
        th += math.pi/2
    return float(ctr[0]), float(ctr[1]), float(wdt), float(hgt), float(th % math.pi)


def iou_axis(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0: return 0.0
    i = (x1-x0)*(y1-y0)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def mat(col):
    return np.stack([np.stack([np.asarray(r, np.float32) for r in m]) for m in col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/obb_labels")
    ap.add_argument("--n-ep", type=int, default=1000)
    ap.add_argument("--check", type=int, default=0, help=">0 이면 그 개수만 YOLO 와 대조")
    a = ap.parse_args()
    import pandas as pd

    corners = pen_corners()
    K, T = cam_high()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ious, nvis = [], 0
    for ep in range(a.check or a.n_ep):
        pq = pathlib.Path(DS) / f"data/chunk-000/episode_{ep:06d}.parquet"
        if not pq.exists(): break
        pen = mat(pd.read_parquet(pq, columns=["pen_pose"])["pen_pose"])
        rows = np.zeros((len(pen), 6), np.float32)       # cx cy w h theta visible
        for i, M in enumerate(pen):
            pw = (M[:3, :3] @ corners.T).T + M[:3, 3]
            uv, z = project(K, T, pw)
            if (z <= 0).any():
                continue
            cx, cy, w_, h_, th = min_area_rect(uv)
            vis = float(0 <= cx < W and 0 <= cy < H)
            rows[i] = (cx/W, cy/H, w_/W, h_/H, th, vis)
            nvis += vis
        np.save(out / f"ep{ep:04d}.npy", rows)

        if a.check:
            y = np.load(f"/workspace/yolo_labels/ep{ep:04d}.npz")["cam_high"]
            for i in range(len(pen)):
                if rows[i, 5] == 0: continue
                p = y[(y[:, 0] == i) & (y[:, 1] == 0)]
                if not len(p): continue
                p = p[p[:, 2].argmax()]
                yb = [p[3]*W, p[4]*H, p[5]*W, p[6]*H]
                cx, cy, w_, h_, th = rows[i, 0]*W, rows[i, 1]*H, rows[i, 2]*W, rows[i, 3]*H, rows[i, 4]
                c, s = math.cos(th), math.sin(th)
                pts = np.array([[cx + dx*c - dy*s, cy + dx*s + dy*c]
                                for dx in (-w_/2, w_/2) for dy in (-h_/2, h_/2)])
                ious.append(iou_axis([pts[:,0].min(), pts[:,1].min(),
                                      pts[:,0].max(), pts[:,1].max()], yb))
        if (ep + 1) % 100 == 0:
            print(f"  {ep+1} 에피소드", flush=True)

    if a.check:
        q = np.array(ious)
        print(f"\nYOLO 축정렬 박스와 대조 ({len(q)} 프레임)")
        print(f"  IoU 중앙값 {np.median(q):.3f} · 평균 {q.mean():.3f} · "
              f"0.5 이상 {np.mean(q>0.5):.0%} · 0.7 이상 {np.mean(q>0.7):.0%}")
    print(f"\n보이는 프레임 {nvis:,}")
    print(f"-> {out}")
    print("OBB-LABELS-DONE")


if __name__ == "__main__":
    main()
