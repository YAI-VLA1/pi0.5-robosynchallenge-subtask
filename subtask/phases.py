#!/usr/bin/env python3
"""복원한 구간에서 프레임별 '위상' 라벨을 만든다.

구간(segment)은 scope 별로 겹친다. 분류기는 단일 라벨이 필요하므로
모든 구간의 시작·끝을 모은 경계로 에피소드를 자른다. 각 조각 안에서는
'실행 중인 구간 집합'이 상수다. items_handover 는 15개가 나온다.
"""
from __future__ import annotations

import json
import pathlib


def phases(schedule_json: pathlib.Path, task: str) -> list[dict]:
    d = json.loads(schedule_json.read_text())[task]
    segs = [(n, s, e) for n, (s, e, _sc) in d["segments"].items()]
    bounds = sorted({x for _, s, e in segs for x in (s, e)})
    out = []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        on = sorted(n for n, s, e in segs if s <= a and e >= b)
        out.append({"id": i, "start": a, "end": b, "segments": on,
                    "name": "+".join(on) if on else "(대기)"})
    return out


def frame_labels(ph: list[dict], n_frames: int) -> list[int]:
    """프레임 -> 위상 id. 스케줄보다 긴 프레임은 마지막 위상으로 본다."""
    lab = [len(ph) - 1] * n_frames
    for p in ph:
        for t in range(p["start"], min(p["end"], n_frames)):
            lab[t] = p["id"]
    return lab


if __name__ == "__main__":
    import sys
    ph = phases(pathlib.Path(sys.argv[1]), sys.argv[2])
    for p in ph:
        print(f"{p['id']:>2}  {p['start']:>4}-{p['end']:<4} ({p['end']-p['start']:>3}f)  {p['name']}")
    print(f"위상 {len(ph)}개")
