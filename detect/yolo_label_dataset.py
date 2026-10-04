#!/usr/bin/env python3
"""학습 데이터 1,000 에피소드 전체에 YOLO11n 박스를 뽑는다.

클래스 0=pen 1=holder 2=distractor (make_yolo_data.py 와 같다).
영상이 AV1 이라 OpenCV 가 못 읽는다 — ffmpeg 로 rawvideo 를 받아 넘긴다.

에피소드당 npz 하나. 이미 있으면 건너뛴다(재개 가능).
  <cam>  : (N,7) float32  [frame, cls, conf, x0, y0, x1, y1]   xyxy 는 0~1 정규화
  meta   : (n_frames, conf_th)
"""
import argparse, os, pathlib, subprocess, sys, time
import numpy as np

DS = pathlib.Path("/workspace/rsc_ws/RoboSynChallenge/policy/pi05/training_data"
                  "/RoboSynChallenge/cobotmagic_Sim_items_handover")
CAMS = {"cam_high": "observation.images.cam_high",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist"}
W, H = 640, 480


def decode(path):
    """mp4 -> (T,H,W,3) uint8 RGB. ffmpeg 가 끝까지 안 읽으면 예외를 던진다."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    buf = p.stdout
    fsz = W * H * 3
    if len(buf) % fsz:
        raise RuntimeError(f"{path}: 프레임 크기 안 맞음 {len(buf)}")
    return np.frombuffer(buf, np.uint8).reshape(-1, H, W, 3)


def run_episode(model, ep, conf_th, batch):
    out = {}
    n_frames = None
    for short, key in CAMS.items():
        vp = DS / "videos" / "chunk-000" / key / f"episode_{ep:06d}.mp4"
        frames = decode(vp)
        if n_frames is None:
            n_frames = len(frames)
        rows = []
        for i in range(0, len(frames), batch):
            chunk = [f for f in frames[i:i + batch]]
            res = model.predict(chunk, imgsz=640, conf=conf_th, device=0,
                                verbose=False, half=True)
            for j, r in enumerate(res):
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy().astype(np.float32)
                xyxy[:, [0, 2]] /= W
                xyxy[:, [1, 3]] /= H
                cls = b.cls.cpu().numpy().astype(np.float32)
                cf = b.conf.cpu().numpy().astype(np.float32)
                fr = np.full_like(cf, i + j)
                rows.append(np.column_stack([fr, cls, cf, xyxy]))
        out[short] = (np.concatenate(rows) if rows
                      else np.zeros((0, 7), np.float32)).astype(np.float32)
    out["meta"] = np.array([n_frames, conf_th], np.float32)
    return out


def worker(rank, eps, args):
    os.environ.setdefault("YOLO_VERBOSE", "false")
    from ultralytics import YOLO
    model = YOLO(args.weights)
    outdir = pathlib.Path(args.out)
    t0, done = time.time(), 0
    for ep in eps:
        dst = outdir / f"ep{ep:04d}.npz"
        if dst.exists():
            continue
        try:
            res = run_episode(model, ep, args.conf, args.batch)
        except Exception as e:                      # 한 에피소드 실패가 전체를 멈추지 않게
            print(f"[w{rank}] ep{ep:04d} 실패: {e}", flush=True)
            continue
        # np.savez 는 .npz 로 안 끝나면 확장자를 덧붙인다 -> 이름을 .npz 로 끝나게 둔다
        tmp = dst.with_name(f"_tmp{rank}_{ep:04d}.npz")
        np.savez_compressed(tmp, **res)
        tmp.rename(dst)
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f"[w{rank}] {done}/{len(eps)} · {el/done:.1f}s/ep · "
                  f"남은 {(len(eps)-done)*el/done/60:.0f}분", flush=True)
    print(f"[w{rank}] 끝 {done}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/yolo_labels")
    ap.add_argument("--weights", default="/root/rsc_recover/bbox_out/yolo/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-ep", type=int, default=1000)
    args = ap.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    todo = [e for e in range(args.n_ep)
            if not (pathlib.Path(args.out) / f"ep{e:04d}.npz").exists()]
    print(f"대상 {len(todo)}/{args.n_ep} 에피소드 · 워커 {args.workers}", flush=True)
    if not todo:
        print("YOLO-LABEL-DONE (이미 전부 있음)")
        return

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for r in range(args.workers):
        sub = todo[r::args.workers]
        if not sub:
            continue
        p = ctx.Process(target=worker, args=(r, sub, args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    have = len(list(pathlib.Path(args.out).glob("ep*.npz")))
    print(f"완료 {have}/{args.n_ep}")
    print("YOLO-LABEL-DONE" if have == args.n_ep else "YOLO-LABEL-INCOMPLETE")


if __name__ == "__main__":
    main()
