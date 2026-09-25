#!/usr/bin/env python3
"""복원한 구간을 물체 궤적으로 검증한다 — 눈이 아니라 숫자로.

sim 영상은 프레임마다 색·재질이 무작위라 육안 대조가 힘들다. 대신 parquet 에
물체의 4x4 pose 가 그대로 들어 있으므로, 물체가 '언제 움직이기 시작하는가'를
구간 경계와 맞춰보면 된다.
"""
from __future__ import annotations

import json
import pathlib
import sys

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

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_R, C_L, C_OBJ = "#e08a5a", "#4a8fd4", "#2f7d52"


def main() -> None:
    task, ep = sys.argv[1], int(sys.argv[2])
    ds = pathlib.Path(sys.argv[3]) / task
    segs = sorted(json.loads(pathlib.Path(sys.argv[4]).read_text())[task]["segments"].items(),
                  key=lambda x: x[1])
    out = pathlib.Path(sys.argv[5])

    f = [p for p in ds.glob("data/**/*.parquet") if f"{ep:06d}" in p.name][0]
    df = pd.read_parquet(f)
    a = np.stack(df["action"].values)
    T = len(a)
    def xyz(col: str) -> np.ndarray:
        # 각 행이 길이 4 의 '배열의 배열' — 4x4 동차변환의 행들. 평행이동은 마지막 열.
        return np.array([[np.asarray(r)[i][3] for i in range(3)] for r in df[col].values])

    pen, hol = xyz("pen_pose"), xyz("holder_pose")
    speed = np.r_[0, np.linalg.norm(np.diff(pen, axis=0), axis=1)]
    d_hol = np.linalg.norm(pen - hol, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(15, 9.2), facecolor=SURF, sharex=True,
                             gridspec_kw=dict(hspace=0.16, height_ratios=[1, 1, .85]))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.885, bottom=0.07)
    for ax in axes:
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.grid(True, axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        for name, (s0, _) in segs:
            c = C_R if name.startswith("r") else C_L
            ax.axvline(s0, color=c, linewidth=1.0, alpha=.45, zorder=0)

    for name, (s0, _) in segs:
        c = C_R if name.startswith("r") else C_L
        axes[0].annotate(name, (s0 + 1, 1.02), xycoords=("data", "axes fraction"), fontsize=7.2,
                         color=c, rotation=38, ha="left", va="bottom", annotation_clip=False)

    axes[0].plot(pen[:, 2], color=C_OBJ, linewidth=2.4, label="펜 z (높이, m)")
    axes[0].set_ylabel("펜 높이", color=INK2, fontsize=10.5)
    axes[1].plot(speed * 1000, color=C_OBJ, linewidth=2.0, label="펜 이동속도 (mm/step)")
    axes[1].set_ylabel("펜 속도", color=INK2, fontsize=10.5)
    axes[2].plot(d_hol, color=C_OBJ, linewidth=2.0, label="펜–홀더 거리 (m)")
    axes[2].plot(a[:, 6], color=C_L, linewidth=1.8, alpha=.8, label="좌그리퍼")
    axes[2].plot(a[:, 13], color=C_R, linewidth=1.8, alpha=.8, linestyle=(0, (4, 2)), label="우그리퍼")
    axes[2].set_ylabel("거리 / 그리퍼", color=INK2, fontsize=10.5)
    axes[2].set_xlabel("timestep", color=INK2, fontsize=10.5)
    for ax in axes:
        lg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
        for x in lg.get_texts():
            x.set_color(INK2)
    axes[0].set_xlim(0, T)

    fig.suptitle(f"{task} 에피소드 {ep} — 복원 구간 vs 물체 궤적 (물체 pose 는 parquet 원본)",
                 color=INK, fontsize=14, fontweight="semibold", x=0.06, ha="left", y=0.975)
    fig.savefig(out, dpi=110, facecolor=SURF)

    mv = np.flatnonzero(speed > 1e-4)
    print(f"펜이 처음 움직인 스텝 {mv[0] if len(mv) else '없음'}, 마지막 {mv[-1] if len(mv) else '-'}")
    print(f"펜 z: 시작 {pen[0,2]:.4f} 최대 {pen[:,2].max():.4f} (스텝 {pen[:,2].argmax()}) 끝 {pen[-1,2]:.4f}")
    print(f"펜-홀더 거리: 시작 {d_hol[0]:.4f} 끝 {d_hol[-1]:.4f}")
    print(f"홀더 이동량 {np.linalg.norm(hol[-1]-hol[0]):.5f} m")
    print(f"저장 {out}")


if __name__ == "__main__":
    main()
