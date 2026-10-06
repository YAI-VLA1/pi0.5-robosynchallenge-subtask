#!/usr/bin/env python3
"""롤아웃 영상에 **모델이 생성한 subtask 지시문**을 매핑해 얹는다.

  python3 subtask_video.py <롤아웃 데이터셋> <subtask 덤프> <출력 폴더> [에피소드...]

매핑
  덤프는 정책 호출마다 한 줄이고 에피소드 경계가 없다. 경계는 데이터셋의
  에피소드 길이에서 복원한다 — 호출 하나가 pi0_step(10) 프레임을 덮으므로
  에피소드 e 의 호출 수는 ceil(len_e / 10) 이다. 합이 덤프 줄 수와 맞는지
  검사하고, 안 맞으면 멈춘다 (조용히 어긋난 라벨을 붙이면 안 된다).
"""
import json, math, pathlib, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PI0_STEP = 10
CAMS = ["observation.images.cam_high", "observation.images.cam_right_wrist"]
W, H = 1280, 480          # 두 뷰 가로 결합
TOP, BAR = 62, 58


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


LOC0, NBIN = 256000, 1024
BBOX_SLOT, BBOX_MID, SUBTASK_SLOT = 4, 4, 14


def parse_dump(dump):
    """덤프 한 줄 -> (문장, 박스 or None).

    bbox 를 켜고 돌린 덤프는 Pen 4 | "; Subtask:" 4 | Subtask 14 = 22 토큰이다.
    끈 덤프는 14 토큰이라 길이로 구분한다.
    """
    import openpi.models.tokenizer as T
    sp = T.PaligemmaTokenizer(200)._tokenizer
    out = []
    for ln in pathlib.Path(dump).read_text().splitlines():
        if not ln.strip():
            continue
        ids = [int(x) for x in ln.split(",")]
        box = None
        if len(ids) >= BBOX_SLOT + BBOX_MID + SUBTASK_SLOT:
            b = ids[:BBOX_SLOT]
            if all(LOC0 <= t < LOC0 + NBIN for t in b):
                ymin, xmin, ymax, xmax = [(t - LOC0) / (NBIN - 1) for t in b]
                box = (xmin, ymin, xmax, ymax)       # 0~1 정규화
            ids = ids[BBOX_SLOT + BBOX_MID:]
        ids = ids[:SUBTASK_SLOT]
        cut = [i for i in ids if i != 0]
        if cut and cut[-1] == 1:
            cut = cut[:-1]
        out.append((sp.decode(cut).strip() or "(빈 슬롯)", box))
    return out


def main():
    root = pathlib.Path(sys.argv[1])
    dump = pathlib.Path(sys.argv[2])
    out  = pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    want = [int(x) for x in sys.argv[4:]] or None

    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines()]
    parsed = parse_dump(dump)
    sents = [x[0] for x in parsed]
    boxes = [x[1] for x in parsed]
    calls = [math.ceil(e["length"] / PI0_STEP) for e in eps]
    # 평가가 아직 돌고 있으면 덤프가 데이터셋보다 앞서거나 뒤처진다.
    # **덤프가 다 들어온 에피소드까지만** 쓴다. 모자란 채로 라벨을 붙이면
    # 경계가 밀려 엉뚱한 지시문이 찍힌다.
    ok_n, acc = 0, 0
    for c in calls:
        if acc + c > len(sents):
            break
        acc += c; ok_n += 1
    if ok_n < len(eps):
        print(f"  (평가 진행 중 — 덤프가 완비된 {ok_n}/{len(eps)} 에피소드만 만든다)")
    eps, calls = eps[:ok_n], calls[:ok_n]
    if not eps:
        sys.exit("★ 아직 완성된 에피소드가 없다")

    meta = {}
    mp = root / "rollout_meta.json"
    if mp.exists():
        meta = {m["episode"]: m for m in json.loads(mp.read_text())}

    f18, f22, f15 = font(18), font(22), font(15)
    off = 0
    for e in eps:
        ep, T_ = e["episode_index"], e["length"]
        n = calls[ep]
        my = sents[off:off + n]
        myb = boxes[off:off + n]
        off += n
        if want is not None and ep not in want:
            continue

        vids = [decode(root / f"videos/chunk-000/{c}/episode_{ep:06d}.mp4") for c in CAMS]
        nfr = min(T_, *(len(v) for v in vids))
        succ = bool(meta.get(ep, {}).get("success", False))

        # 지시문이 바뀌는 지점
        seg = []
        for k, s in enumerate(my):
            if not seg or seg[-1][2] != s:
                seg.append([k * PI0_STEP, k * PI0_STEP, s])
            seg[-1][1] = k * PI0_STEP + PI0_STEP - 1

        frames = []
        for i in range(nfr):
            k = min(i // PI0_STEP, n - 1)
            cur = my[k]
            cv = Image.new("RGB", (W, H + TOP + BAR), (18, 18, 20))
            cv.paste(Image.fromarray(vids[0][i]), (0, TOP))
            cv.paste(Image.fromarray(vids[1][i]), (640, TOP))
            d = ImageDraw.Draw(cv)
            d.text((8, 6), f"ep{ep:03d}   step {i:3d}/{T_-1}   call {k:2d}/{n-1}"
                           f"   {'SUCCESS' if succ else 'fail'}",
                   font=f18, fill=(235, 235, 235))
            d.text((8, 32), f'subtask: "{cur}"', font=f22, fill=(120, 220, 255))
            if myb[k] is None:
                d.text((W - 150, 32), "bbox: 없음", font=f15, fill=(200, 120, 120))
            # 예측 bbox (cam_high 좌표). 모델이 생성한 Pen 슬롯을 그대로 그린다.
            bx = myb[k]
            if bx is not None:
                x0, y0, x1, y1 = bx[0]*640, bx[1]*480, bx[2]*640, bx[3]*480
                for t in range(3):
                    d.rectangle([x0-t, TOP+y0-t, x1+t, TOP+y1+t], outline=(80, 230, 120))
                d.text((max(2, x0), max(TOP, TOP+y0-16)), "pen (pred)", font=f15,
                       fill=(80, 230, 120))
            d.line([640, TOP, 640, TOP + H], fill=(40, 40, 45), width=2)
            d.text((8, TOP + H - 20), "cam_high", font=f15, fill=(230, 230, 60))
            d.text((648, TOP + H - 20), "cam_right_wrist", font=f15, fill=(230, 230, 60))

            # 지시문 타임라인
            y = TOP + H + 16
            x = lambda t: int(8 + (W - 16) * t / max(1, T_ - 1))
            pal = [(70,110,200),(60,150,120),(180,130,50),(150,80,170),
                   (200,70,70),(90,90,160),(60,160,170),(170,110,90)]
            for j, (a, b, s) in enumerate(seg):
                d.rectangle([x(a), y, x(min(b, T_-1)), y + 13], fill=pal[j % len(pal)])
            d.polygon([(x(i), y - 6), (x(i) - 5, y - 14), (x(i) + 5, y - 14)],
                      fill=(255, 255, 255))
            d.text((8, y + 18), f"{len(seg)} subtask segments", font=f15, fill=(190, 190, 195))
            frames.append(np.asarray(cv))

        g = np.stack(frames)
        dst = out / f"ep{ep:03d}_{'success' if succ else 'fail'}_subtask.mp4"
        pr = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{g.shape[2]}x{g.shape[1]}", "-r", "25", "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)],
            stdin=subprocess.PIPE)
        pr.stdin.write(g.tobytes()); pr.stdin.close(); pr.wait()
        print(f"  ep{ep:03d} {nfr} 프레임 · 지시문 {len(seg)}구간 -> {dst.name}", flush=True)
        for a, b, s in seg:
            print(f"      step {a:3d}-{min(b,T_-1):3d}  {s}")
    print(f"\n-> {out}")
    print("SUBTASK-VIDEO-DONE")


if __name__ == "__main__":
    main()
