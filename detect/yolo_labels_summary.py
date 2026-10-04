#!/usr/bin/env python3
"""뽑아 놓은 YOLO 라벨의 커버리지/신뢰도를 요약한다. 학습에 쓰기 전 점검용."""
import pathlib, sys
import numpy as np

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/yolo_labels")
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CLS  = {0: "pen", 1: "holder", 2: "distractor"}

files = sorted(root.glob("ep*.npz"))
print(f"에피소드 {len(files)}\n")
agg = {(c, k): {"cov": [], "conf": [], "n": []} for c in CAMS for k in CLS}
frames = 0
for f in files:
    d = np.load(f)
    T = int(d["meta"][0]); frames += T
    for c in CAMS:
        a = d[c]
        for k in CLS:
            s = a[a[:, 1] == k] if len(a) else a
            agg[(c, k)]["cov"].append(len(np.unique(s[:, 0])) / T if len(s) else 0.0)
            agg[(c, k)]["n"].append(len(s) / T)
            if len(s):
                agg[(c, k)]["conf"].append(float(np.median(s[:, 2])))

print(f"총 프레임 {frames:,} (카메라 3개 => 검출 호출 {frames*3:,})\n")
print(f"{'카메라':16}{'클래스':12}{'프레임 커버리지':>16}{'프레임당 개수':>14}{'conf 중앙값':>12}")
for c in CAMS:
    for k, name in CLS.items():
        g = agg[(c, k)]
        cov = np.mean(g["cov"]); n = np.mean(g["n"])
        cf = np.median(g["conf"]) if g["conf"] else float("nan")
        print(f"{c:16}{name:12}{cov:15.1%}{n:14.2f}{cf:12.3f}")

# pen 을 한 번도 못 찾은 에피소드
bad = []
for f in files:
    d = np.load(f)
    if not len(d["cam_high"]) or (d["cam_high"][:, 1] == 0).sum() == 0:
        bad.append(f.stem)
print(f"\ncam_high 에서 pen 을 한 번도 못 찾은 에피소드 {len(bad)}"
      + (f": {bad[:10]}" if bad else ""))
print(f"\n크기 {sum(f.stat().st_size for f in files)/1e6:.1f} MB")
print("YOLO-SUMMARY-DONE")
