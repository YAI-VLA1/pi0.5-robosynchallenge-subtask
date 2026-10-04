#!/usr/bin/env python3
"""실패 롤아웃에 mistake 라벨과 절단 지점을 붙인다.

왜 stage_metrics 의 t_fail 을 안 쓰나
  subtask 모델은 굳지 않는다 — stage_metrics.json 에서 t_frozen 이 100/100 null 이고
  그래서 t_fail 이 전부 50 으로 고정돼 쓸모가 없다.

그리퍼 규약 (대본 데모에서 확인)
  1 = 열림, 0 = 닫힘. subtask 0 'open both grippers' 구간에 action[13] 이 0 -> 1 로 간다.

실수 시점 t_fail 은 에피소드 종류마다 다르다
  A 잡기 실패 : 오른 그리퍼가 빈 허공에서 닫힌 순간 = t_grasp
  B 놓침      : 잡아서 들었다가(LIFT_DZ 이상이 HOLD 프레임 지속) 다시 떨어진 순간 = t_drop
  C 들고 미완 : 들긴 들었는데 떨어뜨리지도 완수하지도 않음 -> 실수 시점이 특정이 안 된다.
                추측으로 라벨을 만드느니 **제외**한다.
  D 성공      : 자르지 않고 전부 mistake=False.

라벨과 절단 (A, B 공통)
  mistake = True  for t >= t_fail - CHUNK   그 프레임이 내놓는 50-chunk 가 실수를 포함한다
  절단    = [0, t_fail + CHUNK]             마지막 mistake 프레임의 action horizon 을 채운다
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

CHUNK    = 50      # action_horizon
OPEN_TH  = 0.5     # action[13] 이 이보다 크면 열림
CLOSE_TH = 0.5
R_GRIP   = 13      # 오른 그리퍼 축 (좌 6, 우 13)
LIFT_DZ  = 0.03    # 테이블에서 이만큼 뜨면 들린 것
DROP_DZ  = 0.015   # 여기까지 내려오면 놓친 것
HOLD     = 25      # 1초 이상 떠 있어야 '진짜로 들었다' (쳐서 튄 것과 구분)


def grasp_frame(gr_a):
    """처음 열린 뒤 처음 닫히는 프레임."""
    opened = np.flatnonzero(gr_a > OPEN_TH)
    if opened.size == 0:
        return None
    o = int(opened[0])
    closed = np.flatnonzero(gr_a[o:] < CLOSE_TH)
    return int(o + closed[0]) if closed.size else None


def sustained_lift(z, t_grasp):
    """t_grasp 이후에 HOLD 프레임 연속으로 떠 있는 구간의 시작. 없으면 None."""
    up = z > LIFT_DZ
    run = 0
    for t in range(t_grasp, len(z)):
        run = run + 1 if up[t] else 0
        if run >= HOLD:
            return t - HOLD + 1
    return None


def label_episode(ac, pen, success):
    T = len(ac)
    z = pen[:, 2] - pen[0, 2]
    g = grasp_frame(ac[:, R_GRIP])

    def full(reason):
        return {"T": T, "t_grasp": g, "t_fail": None, "t_mistake": None,
                "t_end": T - 1, "n_keep": T, "n_mistake": 0,
                "kind": reason, "use": reason == "success"}

    if success:
        return full("success")
    if g is None:
        return full("no_grasp")

    lift = sustained_lift(z, g)
    if lift is None:
        kind, t_fail = "failed_grasp", g            # A
    else:
        after = np.flatnonzero(z[lift:] < DROP_DZ)
        if after.size == 0:
            return full("lifted_unresolved")        # C: 제외
        kind, t_fail = "dropped", int(lift + after[0])   # B

    t_mis = max(0, t_fail - CHUNK)
    t_end = min(T - 1, t_fail + CHUNK)
    return {"T": T, "t_grasp": g, "t_fail": t_fail, "t_mistake": t_mis,
            "t_end": t_end, "n_keep": t_end + 1, "n_mistake": t_end - t_mis + 1,
            "kind": kind, "use": True}


def main():
    sys.path.insert(0, "/root/rsc_recover")
    import pandas as pd
    from stage_metrics import episode_arrays
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/rollout_ds/subtask")
    meta = {m["episode"]: m for m in json.loads((root / "rollout_meta.json").read_text())}

    rows = []
    for pq in sorted(root.glob("data/chunk-*/episode_*.parquet")):
        ep = int(pq.stem.split("_")[-1])
        st, ac, pen, _ = episode_arrays(pd.read_parquet(pq))
        r = label_episode(ac, pen, bool(meta.get(ep, {}).get("success", False)))
        r["ep"], r["seed"] = ep, meta.get(ep, {}).get("seed")
        rows.append(r)

    (root / "mistake_labels.json").write_text(json.dumps(rows, indent=1))

    kinds = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r)
    names = {"failed_grasp": "A 잡기 실패", "dropped": "B 잡았다 놓침",
             "lifted_unresolved": "C 들고 미완(제외)", "success": "D 성공",
             "no_grasp": "grasp 없음(제외)"}
    print(f"{root.name}: {len(rows)} 에피소드\n")
    for k, v in sorted(kinds.items(), key=lambda x: -len(x[1])):
        eps = "" if len(v) > 8 else "  ep " + ",".join(str(x["ep"]) for x in v)
        print(f"  {names.get(k,k):20} {len(v):4}{eps}")

    use = [r for r in rows if r["use"]]
    fail = [r for r in use if r["t_fail"] is not None]
    print(f"\n  학습에 쓰는 에피소드 {len(use)}/{len(rows)}")
    if fail:
        print(f"\n  {'':16}{'최소':>6}{'25%':>6}{'중앙':>6}{'75%':>6}{'최대':>6}")
        for name, key in (("t_fail", "t_fail"), ("남길 프레임", "n_keep"),
                          ("mistake 프레임", "n_mistake")):
            a = np.array([r[key] for r in fail])
            q = np.percentile(a, [0, 25, 50, 75, 100])
            print(f"  {name:16}" + "".join(f"{v:6.0f}" for v in q))
    keep = sum(r["n_keep"] for r in use)
    mis  = sum(r["n_mistake"] for r in use)
    tot  = sum(r["T"] for r in rows)
    print(f"\n  남는 프레임 {keep:,} / 원본 {tot:,}  (버리는 꼬리 {1-keep/tot:.0%})")
    print(f"  그중 mistake=true {mis:,} = {mis/keep:.0%}")
    print(f"  데모 327,000 과 합치면 전체의 {keep/(327000+keep):.1%}, "
          f"mistake=true 는 {mis/(327000+keep):.1%}")
    print(f"\n-> {root/'mistake_labels.json'}")


if __name__ == "__main__":
    main()
