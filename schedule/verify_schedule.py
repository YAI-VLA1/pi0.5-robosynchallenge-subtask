#!/usr/bin/env python3
"""복원한 subtask 구간이 실제 영상과 맞는지 눈으로 확인한다.

위   구간마다 대표 프레임 (구간 끝 직전 — 그 구간이 '무엇을 끝냈는지' 보인다).
     왼쪽이 cam_high, 오른쪽이 그 구간을 움직이는 팔의 손목 카메라.
     손목 영상이 없으면 cam_high 만 쓴다.
아래 같은 에피소드의 그리퍼 2축 + 구간 경계선. x축을 위 격자와 맞추지는 않는다
     (격자는 균등 배치, 구간 길이는 제각각이라 억지로 맞추면 오히려 헷갈린다).

한계
  프레임 인덱스와 parquet 행 인덱스가 1:1 이라고 가정한다. 둘 다 같은 스텝에서
  기록되므로 맞을 것이나, 확인하지는 않았다.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for f in pathlib.Path("/usr/share/fonts/truetype/nanum").glob("*.ttf"):
    fm.fontManager.addfont(str(f))
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_R, C_L, C_RE, C_LE = "#e08a5a", "#4a8fd4", "#b5502a", "#20578f"   # 우완/좌완/우그리퍼/좌그리퍼


def scope_of(name: str) -> str:
    if "open" in name or "close" in name:
        return "right_eef" if name.startswith("r") or name.startswith("right") else "left_eef"
    return "right_arm" if name.startswith("right") else "left_arm"


COLOR = {"right_arm": C_R, "left_arm": C_L, "right_eef": C_RE, "left_eef": C_LE}


def pick_cam(name: str, cams: dict) -> pathlib.Path | None:
    side = "right" if scope_of(name).startswith("right") else "left"
    return cams.get(f"cam_{side}_wrist")


def grab(video: pathlib.Path, idx: list[int]) -> list | None:
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "f_%03d.png"
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video),
                            "-vf", f"select='{sel}'", "-vsync", "0", str(out)],
                           capture_output=True)
        got = sorted(pathlib.Path(td).glob("f_*.png"))
        if r.returncode or len(got) != len(idx):
            print(f"ffmpeg: rc={r.returncode} 기대 {len(idx)} 장, 받음 {len(got)} 장", flush=True)
            return None
        return [plt.imread(str(p)) for p in got]


def main() -> None:
    task, ep = sys.argv[1], int(sys.argv[2])
    ds = pathlib.Path(sys.argv[3]) / task
    sch = json.loads(pathlib.Path(sys.argv[4]).read_text())[task]
    out = pathlib.Path(sys.argv[5])
    segs = sorted(sch["segments"].items(), key=lambda x: (x[1][0], x[1][1]))

    pq = [p for p in ds.glob("data/**/*.parquet") if f"{ep:06d}" in p.name][0]
    a = np.stack(pd.read_parquet(pq, columns=["action"])["action"].values)
    T = len(a)
    cams = {p.parent.name.split(".")[-1]: p
            for p in ds.glob("videos/**/*.mp4") if f"{ep:06d}" in p.name}
    if not cams:
        sys.exit(f"{task} 에피소드 {ep} 영상 없음")
    high = cams.get("cam_high")

    # 구간 끝 직전 프레임 (그 구간의 결과가 보이는 시점)
    idx = [min(e - 1, T - 1) for _, (s, e) in segs]
    want = sorted(set(idx))
    lut = {}          # (카메라이름, 프레임) -> 이미지
    for cname, cpath in cams.items():
        got = grab(cpath, want)
        if got:
            lut.update({(cname, k): v for k, v in zip(want, got)})
        else:
            print(f"  {cname}: 디코딩 실패", flush=True)

    def panel(name: str, f: int):
        """cam_high 와 해당 팔 손목 카메라를 가로로 붙인다."""
        wr = pick_cam(name, cams)
        parts = [lut.get(("cam_high", f))] if high is not None else []
        if wr is not None:
            parts.append(lut.get((wr.parent.name.split(".")[-1], f)))
        parts = [x for x in parts if x is not None]
        if not parts:
            return None, ""
        h = min(x.shape[0] for x in parts)
        parts = [x[:h] for x in parts]
        lab = "cam_high" + (f" + {pick_cam(name, cams).parent.name.split('.')[-1]}"
                            if len(parts) > 1 else "")
        return np.hstack(parts), lab

    n = len(segs)
    cols = 4
    rows = -(-n // cols)
    fig = plt.figure(figsize=(4.7 * cols, 2.15 * rows + 4.2), facecolor=SURF)
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [1.35],
                          hspace=0.42, wspace=0.06,
                          left=0.045, right=0.985, top=0.935, bottom=0.065)

    for k, (name, (s, e)) in enumerate(segs):
        ax = fig.add_subplot(gs[k // cols, k % cols]); ax.set_facecolor(SURF)
        im, lab = panel(name, min(e - 1, T - 1))
        if im is not None:
            ax.imshow(im)
            ax.text(0.01, 0.985, lab, transform=ax.transAxes, fontsize=7,
                    color="white", ha="left", va="top",
                    bbox=dict(boxstyle="square,pad=0.22", fc="#00000088", ec="none"))
        else:
            ax.text(.5, .5, "프레임 없음", ha="center", va="center", color=MUTED,
                    transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        c = COLOR[scope_of(name)]
        for sp in ax.spines.values():
            sp.set_color(c); sp.set_linewidth(2.4)
        ax.set_title(f"{name}\n{s}–{e}  (t={min(e-1, T-1)})", color=c, fontsize=10,
                     fontweight="semibold", pad=5, linespacing=1.35)

    ax = fig.add_subplot(gs[rows, :]); ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
    t = np.arange(T)
    ax.plot(t, a[:, 6], color=C_LE, linewidth=2.6, label="dim 6 좌그리퍼")
    ax.plot(t, a[:, 13], color=C_RE, linewidth=2.6, label="dim 13 우그리퍼", linestyle=(0, (4, 2)))
    for name, (s, e) in segs:
        c = COLOR[scope_of(name)]
        ax.axvline(s, color=c, linewidth=1.0, alpha=.45, zorder=0)
        ax.annotate(name, (s + 1, 1.045), fontsize=7.2, color=c, rotation=38,
                    ha="left", va="bottom", annotation_clip=False)
    ax.set_xlim(0, T); ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel("timestep", color=INK2, fontsize=10.5)
    ax.set_ylabel("그리퍼 지령 (1=열림, 0=닫힘)", color=INK2, fontsize=10.5)
    ax.tick_params(colors=MUTED, labelsize=9)
    lg = ax.legend(frameon=False, fontsize=9.5, loc="center left")
    for x in lg.get_texts():
        x.set_color(INK2)

    fig.suptitle(f"{task} — 에피소드 {ep}, {T}프레임 · action_config.json 에서 복원한 {n}개 구간"
                 f"  카메라 {len(cams)}개",
                 color=INK, fontsize=14, fontweight="semibold", x=0.045, ha="left", y=0.982)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105, facecolor=SURF)
    print(f"저장 {out}  ({n}구간, 카메라 {sorted(cams)})")


if __name__ == "__main__":
    main()
