#!/usr/bin/env python3
"""action_config.json 의 DAG 를 풀어 에피소드의 subtask 경계를 스텝 단위로 복원한다.

왜
  데이터에 subtask 주석이 없다. 그런데 sim 데이터는 스크립트(키포즈 그래프)로 생성된다.
  스크립트의 edge 마다 duration 이 고정이고 sync 로 순서가 묶여 있으므로,
  경계는 **추정할 필요 없이 계산된다**. 영상 대조도 필요 없다.

검증
  복원한 총 길이가 데이터셋 에피소드 길이와 일치하는지 본다. 일치하면 스케줄이 맞다.
"""
from __future__ import annotations

import json
import pathlib
import sys


def schedule(cfg: dict) -> dict[str, tuple[int, int, str]]:
    """edge 이름 -> (시작, 끝, scope). scope 는 config 가 알려주는 값이지 이름에서 추측한 게 아니다
    (`rclose0` 을 이름으로 판정하면 좌측으로 잘못 간다)."""
    edge, sync = cfg["edge"], cfg.get("sync", {})
    seq, dur, owner = {}, {}, {}
    for sc, lst in edge.items():
        seq[sc] = []
        for item in lst:
            for n, v in item.items():
                seq[sc].append(n)
                dur[n] = v["duration"]
                owner[n] = sc
    start, end = {}, {}
    pend = {sc: list(v) for sc, v in seq.items()}
    for _ in range(1000):
        moved = False
        for sc, q in pend.items():
            while q:
                n = q[0]
                deps = sync.get(n, {}).get("depend_tasks", [])
                if any(x not in end for x in deps):
                    break               # 의존이 아직 안 끝났다 — 이 scope 는 여기서 멈춘다
                s = max([end[x] for x in deps] + [end[m] for m in seq[sc] if m in end] + [0])
                start[n], end[n] = s, s + dur[n]
                q.pop(0)
                moved = True
        if not moved:
            break
    if any(pend.values()):
        raise RuntimeError(f"순환 의존 — 스케줄 못 품: {[q for q in pend.values() if q]}")
    return {n: (start[n], end[n], owner[n]) for n in start}


def main() -> None:
    cfg_root = pathlib.Path(sys.argv[1])       # .../rsc_repo/configs
    ds_root = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out = {}
    for d in sorted(p for p in cfg_root.iterdir() if p.is_dir()):
        p = d / "action_config.json"
        if not p.exists():
            print(f"{d.name:22s} action_config.json 없음 — 건너뜀")
            continue
        try:
            sch = schedule(json.loads(p.read_text()))
        except Exception as e:
            print(f"{d.name:22s} 실패: {type(e).__name__}: {e}")
            continue
        T = max(e for _, e, _ in sch.values())
        real = None
        if ds_root and (ds_root / d.name).exists():
            import pyarrow.parquet as pq
            fs = sorted((ds_root / d.name).glob("data/**/*.parquet"))
            if fs:
                real = pq.ParquetFile(fs[0]).metadata.num_rows
        ok = "일치" if real == T else (f"불일치(데이터 {real})" if real else "데이터 없음")
        print(f"{d.name:22s} 스케줄 {T:>4} 스텝, {len(sch):>2} 구간   {ok}")
        out[d.name] = {"total_steps": T, "dataset_len": real,
                       "scopes": sorted(json.loads(p.read_text())["edge"]),
                       "segments": {n: list(v) for n, v in sorted(sch.items(), key=lambda x: x[1])}}
    pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "schedule.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
