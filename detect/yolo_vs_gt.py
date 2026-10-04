#!/usr/bin/env python3
"""GDINO 의 'pen' 박스가 진짜 펜인지 GT 와 맞춰 센다.

GT 를 어디서 얻나
  10에피소드 실행의 입력 덤프에는 GT 박스가 색으로 그려져 있다
  (pen = 순수 초록 0,255,0 / holder = 순수 빨강 255,0,0).
  그 픽셀 위치를 되읽으면 GT 박스가 정확히 복원된다.

프레임 대응
  입력 덤프는 사이클당 11장(추론 직전 1 + env step 10)이고, 영상은 env step 1:1.
  따라서 호출 i 의 추론 프레임 = 덤프 i*11 = 영상 프레임 i*10.
  탐지는 깨끗한 영상 프레임으로, 채점은 덤프에서 복원한 GT 로 한다.
"""
import pathlib, subprocess, sys, json
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

VID = sys.argv[1]; EPDIR = pathlib.Path(sys.argv[2])
TH = 0.25
dev = "cuda" if torch.cuda.is_available() else "cpu"

tmp = pathlib.Path("/workspace/_gt_frames")
subprocess.run(["rm", "-rf", str(tmp)]); tmp.mkdir(parents=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", VID, str(tmp/"f_%04d.png")], check=True)
vf = sorted(tmp.glob("f_*.png"))
df = sorted(EPDIR.glob("call_*.png"))
print(f"영상 {len(vf)} · 덤프 {len(df)}", flush=True)

def gt_from_dump(p, rgb):
    """cam_high 영역에서 해당 색 픽셀의 경계상자. 없으면 None."""
    a = np.asarray(Image.open(p).convert("RGB"))[:480, :640]
    sel = (a[..., 0] == rgb[0]) & (a[..., 1] == rgb[1]) & (a[..., 2] == rgb[2])
    if not sel.any(): return None
    ys, xs = np.where(sel)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def iou(a, b):
    x0,y0,x1,y1 = max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    if x1<=x0 or y1<=y0: return 0.0
    i=(x1-x0)*(y1-y0)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)

model = YOLO("/workspace/yolo_run/pen_holder/weights/best.pt")
print("YOLO11n 적재", flush=True)

rows = []
for i in range(35):
    vi, di = i*10, i*11
    if vi >= len(vf) or di >= len(df): break
    gt = gt_from_dump(df[di], (0, 255, 0))          # pen = 초록
    crop = Image.open(vf[vi]).convert("RGB").crop((0, 0, 640, 480))
    r = model.predict(crop, conf=TH, verbose=False, device=0)[0]
    pen = sorted([(float(c), [float(v) for v in xy])
                  for c, k, xy in zip(r.boxes.conf, r.boxes.cls, r.boxes.xyxy)
                  if int(k) == 0], reverse=True)   # 0 = pen
    if gt is None:
        rows.append({"call": i, "gt": None, "n": len(pen)}); continue
    ious = [iou(gt, b) for _, b in pen]
    top_ok = bool(ious) and ious[0] >= 0.5
    any_ok = any(v >= 0.5 for v in ious)
    rows.append({"call": i, "gt": gt, "n": len(pen),
                 "top_score": round(pen[0][0], 3) if pen else 0.0,
                 "top_iou": round(ious[0], 3) if ious else 0.0,
                 "best_iou": round(max(ious), 3) if ious else 0.0,
                 "top_ok": top_ok, "any_ok": any_ok,
                 "rank_of_true": (ious.index(max(ious)) + 1) if any_ok else 0})

v = [r for r in rows if r["gt"] is not None]
print(f"\n{'call':>4} {'GT pen':>22} {'박스':>4} {'1등점수':>7} {'1등IoU':>7} {'최고IoU':>7} 진짜펜순위")
for r in v:
    print(f"{r['call']:4} {str(r['gt']):>22} {r['n']:4} {r['top_score']:7.3f} "
          f"{r['top_iou']:7.3f} {r['best_iou']:7.3f} {r['rank_of_true'] or '-':>6}")
n = len(v)
print(f"\nGT 펜이 보이는 호출 {n}/{len(rows)}")
print(f"  1등 박스가 진짜 펜 (IoU>=0.5) : {sum(r['top_ok'] for r in v)}/{n} = {sum(r['top_ok'] for r in v)/n:.0%}")
print(f"  어느 박스든 진짜 펜을 포함    : {sum(r['any_ok'] for r in v)}/{n} = {sum(r['any_ok'] for r in v)/n:.0%}")
print(f"  진짜 펜을 찾았으나 1등이 아님  : {sum(1 for r in v if r['any_ok'] and not r['top_ok'])}/{n}")
print(f"  'pen' 박스 0개                : {sum(1 for r in v if r['n']==0)}/{n}")
pathlib.Path("/workspace/yolo_vs_gt.json").write_text(json.dumps(rows, indent=1))
