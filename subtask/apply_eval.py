#!/usr/bin/env python3
"""학습한 위상 분류기를 평가 롤아웃에 적용해 '시계인가 관측인가'를 가른다.

판별 논리
  데모에서는 위상 = f(timestep) 이 거의 완벽하다(스크립트가 타이밍 고정).
  그래서 데모 정확도는 두 가설을 못 가른다. 평가 롤아웃이 판별한다:

  · 예측이 시계를 따라간다  -> 실패 롤아웃에서도 위상 8~14 가 정시에 나온다.
                              (정책은 실제로 그 단계를 못 했는데도)
  · 예측이 관측을 읽는다    -> 실패 롤아웃에서 위상이 멈추거나 되돌아간다.

  후자면 subtask 예측을 pi0.5 에 넣을 근거가 생긴다. 전자면 넣어도 같은 벽이다.

출력
  · 롤아웃별 예측 위상 시계열 (JSON)
  · 스트립 차트 PNG — 성공/실패 묶음 + 데모 시계 기준선
  · 요약 지표
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from phases import frame_labels
from train_phase import build, normalize

for f in pathlib.Path("/usr/share/fonts/truetype/nanum").glob("*.ttf"):
    fm.fontManager.addfont(str(f))
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def predict(model, arr: np.ndarray, dev: str, batch: int = 256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(arr), batch):
            x = torch.from_numpy(np.ascontiguousarray(arr[i:i + batch])).permute(0, 3, 1, 2).to(dev)
            out.append(model(normalize(x)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-frames", required=True)
    ap.add_argument("--demo-frames", required=True, help="홀드아웃 데모 (통제군)")
    ap.add_argument("--n-demo", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck = torch.load(a.model, map_location="cpu")
    ph = ck["phases"]
    K = len(ph)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build(K).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()

    # ── 통제군: 홀드아웃 데모에서 예측이 시계와 얼마나 맞나 ──
    demo = sorted(pathlib.Path(a.demo_frames).glob("ep*.npy"))[-a.n_demo:]
    d_agree = []
    for f in demo:
        arr = np.load(f, mmap_mode="r")
        p = predict(m, arr, dev)
        clock = np.array(frame_labels(ph, len(arr)))
        d_agree.append(float((p == clock).mean()))
    print(f"통제군(홀드아웃 데모 {len(demo)}개) 시계 일치율 {np.mean(d_agree):.3f}", flush=True)

    # ── 평가 롤아웃 ──
    files = sorted(pathlib.Path(a.eval_frames).glob("*.npy"))
    rows = []
    for f in files:
        arr = np.load(f, mmap_mode="r")
        p = predict(m, arr, dev)
        clock = np.array(frame_labels(ph, len(arr)))
        ok = "SUCCESS" in f.stem
        runmax = np.maximum.accumulate(p)
        rows.append({
            "name": f.stem, "success": ok, "T": int(len(p)),
            "pred": p.tolist(),
            "agree_with_clock": float((p == clock).mean()),
            "max_phase": int(p.max()),
            "reached_left_close": bool((p == 8).any()),
            "reached_left_open1": bool((p == 13).any()),
            "frac_at_or_past_left_close": float((p >= 8).mean()),
            # 예측이 되돌아가는 비율 — 관측 기반이면 실패 시 여기가 커진다
            "regress_frac": float((p < runmax).mean()),
        })
        print(f"  {f.stem:22s} {'성공' if ok else '실패'} "
              f"시계일치 {rows[-1]['agree_with_clock']:.3f} "
              f"최대위상 {rows[-1]['max_phase']:>2} "
              f"되돌아감 {rows[-1]['regress_frac']:.3f}", flush=True)

    S = [r for r in rows if r["success"]]
    F = [r for r in rows if not r["success"]]
    summ = {
        "n_success": len(S), "n_fail": len(F),
        "demo_clock_agreement": float(np.mean(d_agree)),
        "eval_clock_agreement_success": float(np.mean([r["agree_with_clock"] for r in S])) if S else None,
        "eval_clock_agreement_fail": float(np.mean([r["agree_with_clock"] for r in F])) if F else None,
        "fail_reached_left_close": float(np.mean([r["reached_left_close"] for r in F])) if F else None,
        "fail_regress_frac": float(np.mean([r["regress_frac"] for r in F])) if F else None,
        "success_regress_frac": float(np.mean([r["regress_frac"] for r in S])) if S else None,
    }
    print("\n요약")
    for k, v in summ.items():
        print(f"  {k:34s} {v}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summ, "phases": ph, "rollouts": rows, "demo_agreement": d_agree},
              open(out.with_suffix(".json"), "w"), ensure_ascii=False)

    # ── 스트립 차트 ──
    order = S + F
    Tmax = max(r["T"] for r in order)
    M = np.full((len(order) + 2, Tmax), np.nan)
    M[0, :] = np.array(frame_labels(ph, Tmax))          # 데모 시계 기준선
    for i, r in enumerate(order):
        M[i + 2, :r["T"]] = r["pred"]
    cmap = ListedColormap(plt.cm.turbo(np.linspace(0.04, 0.96, K)))
    cmap.set_bad(SURF)
    fig, ax = plt.subplots(figsize=(15, max(5.0, 0.16 * len(order) + 2.6)), facecolor=SURF)
    fig.subplots_adjust(left=0.135, right=0.9, top=0.88, bottom=0.08)
    ax.set_facecolor(SURF)
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap,
                   norm=BoundaryNorm(np.arange(-0.5, K + 0.5), K))
    ax.set_yticks([0] + [i + 2 for i in range(len(order))])
    ax.set_yticklabels(["데모 시계 (기준)"] + [r["name"].replace("ep", "") for r in order], fontsize=6.6)
    for t in ax.get_yticklabels()[1:]:
        t.set_color(INK2)
    ax.get_yticklabels()[0].set_color(INK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    if S:
        ax.axhline(len(S) + 1.5, color=INK, linewidth=1.2)
        ax.text(Tmax * 1.005, 2 + len(S) / 2, f"성공 {len(S)}", rotation=90, va="center",
                fontsize=9, color=INK2)
        ax.text(Tmax * 1.005, 2 + len(S) + len(F) / 2, f"실패 {len(F)}", rotation=90, va="center",
                fontsize=9, color=INK2)
    ax.set_xlabel("환경 스텝", color=INK2, fontsize=10.5)
    ax.tick_params(colors=MUTED, labelsize=9)
    cb = fig.colorbar(im, ax=ax, ticks=range(K), pad=0.035, fraction=0.028)
    cb.ax.set_yticklabels([f"{p['id']} {p['name'][:26]}" for p in ph], fontsize=6.6, color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("평가 롤아웃의 예측 위상 — 맨 위가 데모의 고정 스케줄(시계)",
                 color=INK, fontsize=13, fontweight="semibold", loc="left", pad=12)
    fig.savefig(out.with_suffix(".png"), dpi=115, facecolor=SURF)
    print(f"\n저장 {out.with_suffix('.json')} · {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
