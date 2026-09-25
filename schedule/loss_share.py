#!/usr/bin/env python3
"""학습 손실에서 각 action 차원이 실제로 차지하는 몫을 잰다.

손실은 32차원 평균이다(14 실축 + 18 제로패딩). 패딩 축을 빼면 무엇이 달라지는지,
그리퍼 축이 얼마나 작은 몫인지 숫자로 본다.

방법
  학습 파이프라인과 같은 변환을 재현한다: 청크 기준 delta(팔) + 절대(그리퍼)
  → q01/q99 분위 정규화. 그 뒤 축별 분산을 본다. 흐름매칭 손실의 축별 잔차는
  조건부 분산에 비례하므로, 분산 몫이 손실 몫의 상한이 된다.

한계
  q01/q99 는 여기 표본(20 에피소드)에서 다시 계산한다. 체크포인트의 norm_stats 와
  완전히 같지는 않다. 비율을 보는 용도로만 쓴다.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

H = 50          # action_horizon
ARM = list(range(0, 6)) + list(range(7, 13))
GRIP = [6, 13]


def main() -> None:
    ds = pathlib.Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    fs = sorted(ds.glob("data/**/*.parquet"))[:n]
    tgt = []
    for f in fs:
        df = pd.read_parquet(f, columns=["action", "observation.state"])
        a = np.stack(df["action"].values).astype(np.float64)
        s = np.stack(df["observation.state"].values).astype(np.float64)
        T = len(a)
        for t0 in range(0, T - H):
            ch = a[t0:t0 + H].copy()
            ch[:, ARM] -= s[t0, ARM]        # 팔은 청크 시작 상태 기준 delta
            tgt.append(ch)
    X = np.concatenate(tgt)                 # (N, 14)
    q01, q99 = np.quantile(X, .01, axis=0), np.quantile(X, .99, axis=0)
    Z = (X - q01) / (q99 - q01 + 1e-6) * 2 - 1

    v = Z.var(axis=0)
    tot = v.sum()
    print(f"표본 {len(fs)} 에피소드, 타깃 {len(X):,} × 14\n")
    print(f"{'dim':>4} {'분산':>9} {'14축 중 몫':>11} {'32축 손실 중 몫':>15}")
    for i in range(14):
        tag = " ← 그리퍼" if i in GRIP else ""
        print(f"{i:>4} {v[i]:9.4f} {v[i]/tot*100:10.2f}% {v[i]/tot*100*14/32:14.2f}%{tag}")
    g = sum(v[i] for i in GRIP)
    print(f"\n그리퍼 2축 합: 14축 중 {g/tot*100:.1f}%  ·  32축 손실 중 {g/tot*100*14/32:.1f}%")
    print(f"좌그리퍼(dim6) 단독: 14축 중 {v[6]/tot*100:.1f}%  ·  32축 손실 중 {v[6]/tot*100*14/32:.1f}%")
    print(f"\n패딩 18축을 마스킹하면 모든 축의 몫이 일률적으로 {32/14:.2f}배 커진다 —")
    print("축 사이의 상대 비중은 바뀌지 않는다.")


if __name__ == "__main__":
    main()
