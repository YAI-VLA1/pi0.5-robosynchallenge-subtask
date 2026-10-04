#!/usr/bin/env python3
"""롤아웃을 단계별로 채점하고 실패 시점을 찾는다.

왜
  전체 성공률이 0~1% 라 중간 개선이 전부 가려진다. 어느 단계까지 갔는지를
  따로 재야 모델 비교가 된다. 같은 계산이 '실수가 언제 일어났나'도 주고,
  그게 50-chunk mistake 라벨과 데이터 절단의 기준이 된다.

parquet 를 직접 읽는다 — LeRobotDataset 인덱싱은 프레임마다 영상을 디코드해 수백 배 느리다.
판정은 pen_pose / holder_pose / qpos 에서만 나온다. 사람 라벨링 없음.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

LIFT_DZ   = 0.03    # 테이블에서 이만큼 뜨면 들린 것
CENTER_DY = 0.15    # 시작 y 에서 이만큼 좌측으로 오면 중앙 이동
NEAR_XY   = 0.06    # holder 중심에서 이 안이면 위에 있는 것
PLACE_DZ  = 0.12    # holder 높이 + 이 안으로 내려오면 꽂힌 것
CMD_EPS   = 0.010   # |action - state| 가 이보다 작으면 '움직이라는 명령이 없다'
MOVE_EPS  = 0.005   # 실제 이동이 이보다 작으면 '안 움직였다'
STUCK_WIN = 25      # 25 프레임(1초) 이상 끝까지 정지 명령이면 굳은 것
FAIL_OFFSET = 50    # 명령이 멈춘 뒤에도 팔은 목표 자세까지 더 간다. 영상으로 보면
                    # 마커가 일찍 찍혀 보여서 뒤로 민다 (2026-10-04 눈으로 확인)
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]   # 그리퍼(6,13) 제외한 팔 축


def _mat(col):
    """4x4 pose 는 행 4개가 object 배열로 중첩돼 저장된다. 평평하게 펴서 (T,3) 위치만."""
    return np.stack([np.stack([np.asarray(r, dtype=np.float32) for r in m])
                     for m in col])[:, :3, 3]


def episode_arrays(df):
    st  = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    ac  = np.stack(df["action"].to_numpy()).astype(np.float32)
    return st, ac, _mat(df["pen_pose"]), _mat(df["holder_pose"])


def analyse(st, ac, pen, hol):
    T = len(pen)
    z0, y0, H = pen[0, 2], pen[0, 1], hol[0]
    out = {"T": T}

    def first(c):
        i = np.flatnonzero(c)
        return int(i[0]) if i.size else None

    out["t_lift"]   = first(pen[:, 2] > z0 + LIFT_DZ)
    out["t_center"] = first(pen[:, 1] > y0 + CENTER_DY)
    over = (np.abs(pen[:, 0] - H[0]) < NEAR_XY) & (np.abs(pen[:, 1] - H[1]) < NEAR_XY)
    out["t_over"]   = first(over)
    out["t_place"]  = first(over & (pen[:, 2] < H[2] + PLACE_DZ))

    stage = 0
    for k, key in enumerate(("t_lift", "t_center", "t_over", "t_place"), start=1):
        if out[key] is not None:
            stage = k
    out["stage"] = stage

    # action 은 절대 qpos 다 (|a[t]-s[t]| 중앙값 0.0001). 따라서
    #   |a[t]-s[t]| 가 작다  = 움직이라는 명령 자체가 없다 = 정책이 멈춘 것
    #   명령은 큰데 안 움직인다 = 물리적으로 막힌 것 (실측 0.9% 로 드물다)
    cmd   = np.abs(ac[:, ARM] - st[:, ARM]).max(1)
    moved = np.r_[np.abs(np.diff(st[:, ARM], axis=0)).max(1), 0.0]
    still   = cmd < CMD_EPS
    blocked = (cmd >= CMD_EPS) & (moved < MOVE_EPS)
    out["blocked_frac"] = round(float(blocked.mean()), 4)
    runs, cur = [], 0
    for s in still:
        cur = cur + 1 if s else 0
        runs.append(cur)
    runs = np.array(runs)
    out["stuck_frac"] = round(float(still.mean()), 4)   # = 정지 명령 비율
    out["stuck_max"]  = int(runs.max())
    moved = np.flatnonzero(~still)
    out["t_frozen"] = int(moved[-1]) if moved.size and (T - moved[-1]) >= STUCK_WIN else None

    reached = [out[k] for k in ("t_lift", "t_center", "t_over", "t_place") if out[k] is not None]
    base = out["t_frozen"] if out["t_frozen"] is not None else (
        max(reached) if reached else 0)
    out["t_stop"] = base                       # 명령이 멈춘 시점 (원값)
    out["t_fail"] = min(T - 1, base + FAIL_OFFSET)   # 실제 실패로 볼 시점
    return out


def main():
    root = pathlib.Path(sys.argv[1])
    import pandas as pd
    meta = {m["episode"]: m for m in json.loads((root / "rollout_meta.json").read_text())}
    rows = []
    for pq in sorted(root.glob("data/chunk-*/episode_*.parquet")):
        ep = int(pq.stem.split("_")[-1])
        st, ac, pen, hol = episode_arrays(pd.read_parquet(pq))
        r = analyse(st, ac, pen, hol)
        m = meta.get(ep, {})
        r.update(ep=ep, seed=m.get("seed"), success=bool(m.get("success", False)))
        rows.append(r)
    if not rows:
        sys.exit("에피소드가 없다")

    names = ["아무것도", "1 들기", "2 중앙이동", "3 holder위", "4 삽입"]
    hist = np.bincount([r["stage"] for r in rows], minlength=5)
    print(f"{root.name}: {len(rows)} 에피소드\n")
    print(f"{'도달 단계':14} {'수':>4} {'비율':>7}")
    for i, n in enumerate(hist):
        print(f"  {names[i]:12} {n:4} {n/len(rows):7.0%}")
    fr = [r for r in rows if r["t_frozen"] is not None]
    print(f"\n성공(공식)        {sum(r['success'] for r in rows)}/{len(rows)}")
    print(f"굳어버림          {len(fr)}/{len(rows)} = {len(fr)/len(rows):.0%}")
    print(f"정지 명령 비율     중앙값 {np.median([r['stuck_frac'] for r in rows]):.1%}")
    print(f"물리적 막힘 비율   중앙값 {np.median([r['blocked_frac'] for r in rows]):.1%}")
    print(f"최장 정지 구간     중앙값 {np.median([r['stuck_max'] for r in rows]):.0f} 프레임")

    tf = np.array([r["t_fail"] for r in rows])
    ts = np.array([r["t_stop"] for r in rows])
    print(f"\n명령 정지 t_stop  중앙값 {np.median(ts):.0f}  (t_fail = t_stop + {FAIL_OFFSET})")
    T  = int(np.median([r["T"] for r in rows]))
    print(f"\n=== t_fail 분포 (에피소드 길이 {T}) ===")
    for q in (0, 10, 25, 50, 75, 90, 100):
        print(f"  {q:3}%  {np.percentile(tf, q):6.0f}")
    print(f"  평균 {tf.mean():.0f} · 표준편차 {tf.std():.0f}")
    print(f"\n실패 후 남는 꼬리 (T - t_fail) 중앙값 {np.median(T - tf):.0f} 프레임")
    (root / "stage_metrics.json").write_text(json.dumps(rows, indent=1))
    print(f"\n-> {root/'stage_metrics.json'}")


if __name__ == "__main__":
    main()
