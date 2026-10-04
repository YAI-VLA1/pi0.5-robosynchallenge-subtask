#!/usr/bin/env python3
"""롤아웃 영상에 단계 판정과 실패 시점을 입힌다.

stage_metrics.json 의 t_lift / t_center / t_over / t_place / t_fail 을 타임라인으로
그리고, 현재 프레임이 어느 단계인지와 정지 여부를 하단에 표시한다.
"""
import json, pathlib, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DS = pathlib.Path(sys.argv[1])          # rollout_ds/<name>
OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
EPS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]

F  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
STAGES = [("t_lift", "1 lift", "#00E676"), ("t_center", "2 to center", "#2A78D6"),
          ("t_over", "3 over holder", "#AF52DE"), ("t_place", "4 placed", "#FFD60A")]
CAMS = ["observation.images.cam_high", "observation.images.cam_right_wrist",
        "observation.images.cam_left_wrist"]

rows = {r["ep"]: r for r in json.loads((DS / "stage_metrics.json").read_text())}

for ep in EPS:
    r = rows.get(ep)
    if r is None:
        print(f"ep{ep}: 지표 없음"); continue
    vids = [DS / f"videos/chunk-000/{c}/episode_{ep:06d}.mp4" for c in CAMS]
    if not all(v.exists() for v in vids):
        print(f"ep{ep}: 영상 없음"); continue
    import pandas as pd
    _pq = DS / f"data/chunk-000/episode_{ep:06d}.parquet"
    GRIP = None
    if _pq.exists():
        _st = np.stack(pd.read_parquet(_pq)["observation.state"].to_numpy()).astype(np.float32)
        GRIP = [(float(x[13]), float(x[6])) for x in _st]
    tmp = pathlib.Path(f"/workspace/_sv{ep}"); subprocess.run(["rm","-rf",str(tmp)]); tmp.mkdir()
    for i, v in enumerate(vids):
        (tmp / f"c{i}").mkdir()
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(v),
                        str(tmp/f"c{i}"/"f_%04d.png")], check=True)
    frames = [sorted((tmp/f"c{i}").glob("f_*.png")) for i in range(3)]
    T = min(len(f) for f in frames)
    out = tmp/"o"; out.mkdir()
    W, BAR = 1920, 96
    for t in range(T):
        im = Image.new("RGB", (W, 480 + BAR), "#111111")
        for i in range(3):
            im.paste(Image.open(frames[i][t]).convert("RGB"), (i*640, 0))
        d = ImageDraw.Draw(im)
        # 타임라인
        y0, x0, xw = 496, 20, W - 240
        d.rectangle([x0, y0, x0+xw, y0+8], fill="#333333")
        d.rectangle([x0, y0, x0 + int(xw*t/max(1,T-1)), y0+8], fill="#888888")
        for key, label, col in STAGES:
            v = r.get(key)
            if v is None: continue
            x = x0 + int(xw * v / max(1, T-1))
            d.rectangle([x-2, y0-7, x+2, y0+15], fill=col)
            d.text((x-4, y0-26), label, fill=col, font=FS)
        tf = r.get("t_fail")
        if tf is not None:
            x = x0 + int(xw * tf / max(1, T-1))
            d.rectangle([x-2, y0-14, x+2, y0+22], fill="#FF3B30")
            d.text((x-4, y0+26), f"t_fail {tf}", fill="#FF3B30", font=FS)
        # 현재 단계
        cur = 0
        for k, (key, _, _) in enumerate(STAGES, start=1):
            if r.get(key) is not None and t >= r[key]: cur = k
        names = ["-", "lifted", "centered", "over holder", "placed"]
        d.text((20, y0+22), f"ep{ep:02d}  stage {cur} {names[cur]}", fill="white", font=FB)
        if GRIP is not None and t < len(GRIP):
            rg, lg = GRIP[t]
            d.text((420, y0+24), f"R grip {rg:.2f} " + ("CLOSED" if rg < 0.5 else "open"),
                   fill="#00E676" if rg < 0.5 else "#9AA0A6", font=F)
            d.text((680, y0+24), f"L grip {lg:.2f} " + ("CLOSED" if lg < 0.5 else "open"),
                   fill="#00E676" if lg < 0.5 else "#FF3B30", font=F)
        d.text((W-210, y0-2), f"frame {t:3}/{T}", fill="#9AA0A6", font=F)
        if tf is not None and t >= tf:
            d.text((W-210, y0+26), "FROZEN" if r.get("t_frozen") is not None else "FAILED",
                   fill="#FF3B30", font=F)
        im.save(out/f"o_{t:04d}.png")
    dst = OUT / f"ep{ep:02d}_stage.mp4"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","25","-i",str(out/"o_%04d.png"),
                    "-c:v","libx264","-pix_fmt","yuv420p","-crf","20",str(dst)], check=True)
    subprocess.run(["rm","-rf",str(tmp)])
    print(f"-> {dst}  (stage {r['stage']}, t_fail {r.get('t_fail')}, stuck {r['stuck_frac']:.0%})")
