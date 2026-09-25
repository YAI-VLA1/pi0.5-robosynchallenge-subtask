#!/usr/bin/env python3
"""영상을 위상 분류기 학습용 uint8 배열로 디코딩한다.

에피소드 하나 -> (T, H, 3*W, 3) uint8. 세 카메라를 가로로 붙인다.
순서는 [cam_high, cam_right_wrist, cam_left_wrist] — 평가 롤아웃 영상의 패널 순서와 같다
(`img_arr=[front,right,left]`, 2026-09-23 확인).

데모(개별 mp4 3개)와 평가(1920x480 3분할 mp4 1개)를 같은 형식으로 만든다.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np

SIDE = 112
CAMS = ["cam_high", "cam_right_wrist", "cam_left_wrist"]


def raw(path: pathlib.Path, w: int, h: int, vf: str) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf,
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True)
    if p.returncode:
        raise RuntimeError(f"{path.name}: ffmpeg {p.stderr.decode()[:200]}")
    a = np.frombuffer(p.stdout, np.uint8)
    n = w * h * 3
    if len(a) % n:
        raise RuntimeError(f"{path.name}: 프레임 크기 불일치 {len(a)} % {n}")
    return a.reshape(-1, h, w, 3)


def demo_files(root: pathlib.Path, ep: int) -> list[pathlib.Path]:
    return [root / f"videos/chunk-000/observation.images.{c}/episode_{ep:06d}.mp4" for c in CAMS]


def demo_episode(root: pathlib.Path, ep: int) -> np.ndarray:
    views = []
    for f in demo_files(root, ep):
        views.append(raw(f, SIDE, SIDE, f"scale={SIDE}:{SIDE}"))
    n = min(len(v) for v in views)
    if len({len(v) for v in views}) > 1:
        print(f"  ep{ep}: 카메라별 프레임 수가 다르다 {[len(v) for v in views]} -> {n} 로 자름")
    return np.concatenate([v[:n] for v in views], axis=2)


def eval_episode(f: pathlib.Path) -> np.ndarray:
    """1920x480 = 640x480 패널 3개. 각각 정사각으로 리사이즈해 데모와 같은 형식으로."""
    a = raw(f, 1920, 480, "scale=1920:480")
    out = np.empty((len(a), SIDE, SIDE * 3, 3), np.uint8)
    from PIL import Image
    for i, fr in enumerate(a):
        for k in range(3):
            panel = Image.fromarray(fr[:, k * 640:(k + 1) * 640]).resize((SIDE, SIDE), Image.BILINEAR)
            out[i, :, k * SIDE:(k + 1) * SIDE] = np.asarray(panel)
    return out


def main() -> None:
    mode, src, dst = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    dst.mkdir(parents=True, exist_ok=True)
    if mode == "demo":
        eps = sorted({int(p.stem.split("_")[1])
                      for p in src.glob("videos/chunk-000/observation.images.cam_high/*.mp4")})
        for ep in eps:
            o = dst / f"ep{ep:06d}.npy"
            if o.exists():
                continue
            # 세 카메라가 다 와야 의미가 있다. 하나라도 없으면 디코딩을 시작조차 하지 않는다
            # (cam_high 만 먼저 받아지는 구간에서 매 회차 몇 분씩 버리게 된다).
            if not all(f.exists() and f.stat().st_size > 0 for f in demo_files(src, ep)):
                continue
            try:
                np.save(o, demo_episode(src, ep))
            except Exception as e:
                print(f"  ep{ep}: 실패 {e}", flush=True)
                continue
            if ep % 20 == 0:
                print(f"  ep{ep} 완료", flush=True)
        print(f"데모 {len(list(dst.glob('*.npy')))} 에피소드", flush=True)
    else:
        for f in sorted(src.glob("*.mp4")):
            o = dst / (f.stem + ".npy")
            if o.exists():
                continue
            try:
                np.save(o, eval_episode(f))
                print(f"  {f.stem} 완료", flush=True)
            except Exception as e:
                print(f"  {f.stem}: 실패 {e}", flush=True)
        print(f"평가 {len(list(dst.glob('*.npy')))} 롤아웃", flush=True)


if __name__ == "__main__":
    main()
