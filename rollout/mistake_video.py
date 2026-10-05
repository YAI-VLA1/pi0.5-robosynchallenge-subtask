#!/usr/bin/env python3
"""mistake=true 구간을 눈으로 보게 영상에 표시한다.

  · 빨간 테두리  = mistake: true 인 프레임
  · 노란 세로선  = t_fail (실수 시점)
  · 아래 타임바  = 에피소드 전체 중 어디인지. 빨간 구간이 mistake, 회색이 잘려 나간 꼬리
"""
import json, pathlib, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROLL = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else "/workspace/rollout_ds/subtask_aligned")
OUT  = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/mistake_vid")
CAMS = ["observation.images.cam_high", "observation.images.cam_right_wrist"]
W, H = 1280, 480     # 두 카메라를 가로로 붙인다
BAR  = 56          # 아래 타임바 높이
TOP  = 34          # 위 제목줄

# 이 컨테이너에는 한글 폰트가 없다 (dejavu 만 있다). 한글을 쓰면 네모로 나온다.
KIND_EN = {"failed_grasp": "A  gripper closed on empty air",
           "dropped": "B  grasped then dropped",
           "success": "D  success",
           "lifted_unresolved": "C  lifted, unresolved (EXCLUDED)"}


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if pathlib.Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def decode(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, 480, 640, 3)


def decode_pair(ep):
    """cam_high | cam_right_wrist 를 가로로 붙인다. 집는 순간은 손목 카메라에서 보인다."""
    a = [decode(ROLL / f"videos/chunk-000/{c}/episode_{ep:06d}.mp4") for c in CAMS]
    n = min(len(x) for x in a)
    return np.concatenate([a[0][:n], a[1][:n]], axis=2)


def render(ep, lab, frames):
    """에피소드 하나를 (T, H+TOP+BAR, W, 3) 로 그린다."""
    T = lab["T"]
    t_mis, t_fail, t_end = lab["t_mistake"], lab["t_fail"], lab["t_end"]
    f14, f20, f26 = font(14), font(20), font(15)
    out = []
    n = min(len(frames), T)
    for i in range(n):
        mis = t_mis is not None and t_mis <= i <= t_end
        cut = t_end is not None and i > t_end          # 학습 데이터에서 잘려 나간 구간
        canvas = Image.new("RGB", (W, H + TOP + BAR), (18, 18, 20))
        canvas.paste(Image.fromarray(frames[i]), (0, TOP))
        d = ImageDraw.Draw(canvas)

        # 제목
        kind = KIND_EN.get(lab["kind"], lab["kind"])
        d.text((8, 7), f"ep{ep:03d}  {kind}", font=f14, fill=(235, 235, 235))
        if cut:
            tag, col = "CUT  (not in training set)", (150, 150, 155)
        elif mis:
            tag, col = "mistake: TRUE", (255, 70, 70)
        else:
            tag, col = "mistake: false", (120, 200, 120)
        d.text((W - 240, 7), tag, font=f14, fill=col)

        # mistake 프레임은 테두리
        if mis or cut:
            bc = (255, 50, 50) if mis else (90, 90, 95)
            for k in range(4):
                d.rectangle([k, TOP + k, W - 1 - k, TOP + H - 1 - k], outline=bc)
        # 두 카메라 경계
        d.line([640, TOP, 640, TOP + H], fill=(40, 40, 45), width=2)
        d.text((8, TOP + H - 20), "cam_high", font=f14, fill=(230, 230, 60))
        d.text((648, TOP + H - 20), "cam_right_wrist", font=f14, fill=(230, 230, 60))

        # 아래 타임바
        y0 = TOP + H + 14
        x = lambda t: int(8 + (W - 16) * t / max(1, T - 1))
        d.rectangle([8, y0, W - 8, y0 + 12], fill=(60, 60, 64))             # 전체
        if t_end is not None and t_end < T - 1:                              # 버린 꼬리
            d.rectangle([x(t_end), y0, W - 8, y0 + 12], fill=(38, 38, 40))
        if t_mis is not None:                                                # mistake 구간
            d.rectangle([x(t_mis), y0, x(t_end), y0 + 12], fill=(190, 45, 45))
        if t_fail is not None:                                               # 실수 시점
            d.line([x(t_fail), y0 - 5, x(t_fail), y0 + 17], fill=(255, 215, 0), width=3)
        d.polygon([(x(i), y0 - 7), (x(i) - 5, y0 - 15), (x(i) + 5, y0 - 15)],
                  fill=(255, 255, 255))                                      # 현재 위치
        lb = f"frame {i:3d}/{T-1}"
        if t_fail is not None:
            lb += (f"    t_fail {t_fail}    mistake window {t_mis}-{t_end}"
                   f"    (tail dropped: {T-1-t_end} frames)")
        else:
            lb += "    no mistake  (full episode kept)"
        d.text((8, y0 + 18), lb, font=f26, fill=(200, 200, 205))
        out.append(np.asarray(canvas))
    return np.stack(out)


def write(frames, path, fps=25):
    h, w = frames.shape[1:3]
    pr = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
        stdin=subprocess.PIPE)
    pr.stdin.write(frames.astype(np.uint8).tobytes())
    pr.stdin.close()
    pr.wait()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labs = {r["ep"]: r for r in json.loads((ROLL / "mistake_labels.json").read_text())}

    # A 4개 + B 2개 + D 1개 + C(제외) 1개
    pick, by = [], {}
    for r in labs.values():
        by.setdefault(r["kind"], []).append(r["ep"])
    for kind, k in (("failed_grasp", 4), ("dropped", 2),
                    ("success", 1), ("lifted_unresolved", 1)):
        pick += sorted(by.get(kind, []))[:k]
    print("대상 에피소드", pick)

    clips = {}
    for ep in pick:
        clip = render(ep, labs[ep], decode_pair(ep))
        clips[ep] = clip
        write(clip, OUT / f"ep{ep:03d}_{labs[ep]['kind']}.mp4")
        print(f"  ep{ep:03d} {labs[ep]['kind']:18} {len(clip)} 프레임 -> ep{ep:03d}_{labs[ep]['kind']}.mp4")

    # 2x3 그리드 (A 4개 + B 2개)
    grid_eps = [e for e in pick if labs[e]["kind"] in ("failed_grasp", "dropped")][:4]
    if len(grid_eps) == 4:
        n = max(len(clips[e]) for e in grid_eps)
        ch, cw = clips[grid_eps[0]].shape[1:3]
        g = np.zeros((n, ch * 2, cw * 2, 3), np.uint8)
        for k, e in enumerate(grid_eps):
            c = clips[e]
            pad = np.repeat(c[-1:], n - len(c), axis=0) if len(c) < n else c[:0]
            c = np.concatenate([c, pad]) if len(pad) else c
            r, col = divmod(k, 2)
            g[:, r*ch:(r+1)*ch, col*cw:(col+1)*cw] = c
        write(g, OUT / "grid_4.mp4")
        print(f"  그리드 2x2 -> grid_4.mp4 ({n} 프레임, {cw*2}x{ch*2})")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
