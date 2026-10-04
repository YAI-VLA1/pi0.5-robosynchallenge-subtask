#!/usr/bin/env python3
"""이미 수집한 롤아웃의 (obs, action) 한 칸 어긋남을 복구한다. (리뷰 R1)

저장된 행 i 는  obs = o_{i+1},  action = a_i  다 (a_i 는 o_i 에서 나왔다).
행 i 의 올바른 짝은 a_{i+1} 이므로 **action 열만 한 칸 당기고 마지막 행을 버린다**.

  before  row i : (o_{i+1}, a_i)
  after   row i : (o_{i+1}, a_{i+1})   for i = 0 .. N-2

잃는 것은 (o_0, a_0) 한 쌍이다 — o_0(리셋 직후)은 애초에 기록되지 않았다.
영상은 건드리지 않는다. LeRobot 은 parquet 길이만큼만 프레임을 요청한다.

원본은 그대로 두고 새 디렉터리에 쓴다.
"""
from __future__ import annotations
import json, pathlib, shutil, sys

import numpy as np
import pandas as pd

CAMS = ["observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]


def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/rollout_ds/subtask")
    dst = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else str(src) + "_aligned")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    info = json.loads((src / "meta/info.json").read_text())
    # 이중 shift 방지. 한 번 더 밀면 (o_{t+1}, a_{t+2}) 가 되어 조용히 더 나빠진다.
    if info.get("rsc_alignment_fixed"):
        sys.exit(f"★ {src} 는 이미 정렬 복구된 데이터다 (meta/info.json 의 "
                 f"rsc_alignment_fixed). 원본(raw) 을 넘길 것.")
    eps  = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (src / "meta/episodes.jsonl").read_text().splitlines()}

    running = 0
    out_eps = []
    for pq in sorted(src.glob("data/chunk-*/episode_*.parquet")):
        ep = int(pq.stem.split("_")[-1])
        df = pd.read_parquet(pq)
        n = len(df)
        assert n >= 2, f"ep{ep}: 프레임 {n} 개로는 못 민다"

        ac = np.stack(df["action"].to_numpy())
        df = df.iloc[:-1].copy()                       # 마지막 행은 짝이 없다
        df["action"] = list(ac[1:])                    # action 만 한 칸 당긴다
        df["frame_index"] = np.arange(len(df), dtype=np.int64)
        df["index"] = np.arange(running, running + len(df), dtype=np.int64)
        df["timestamp"] = (np.arange(len(df)) / info["fps"]).astype(np.float32)
        running += len(df)

        out = dst / pq.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)

        e = dict(eps[ep]); e["length"] = len(df)
        out_eps.append(e)

    # 영상·나머지 meta 는 링크/복사
    for cam in CAMS:
        for mp4 in sorted((src / "videos/chunk-000" / cam).glob("*.mp4")):
            d = dst / mp4.relative_to(src)
            d.parent.mkdir(parents=True, exist_ok=True)
            d.symlink_to(mp4.resolve())
    for f in ("meta/tasks.jsonl", "meta/episodes_stats.jsonl"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    for f in ("rollout_meta.json",):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    info = dict(info)
    info["rsc_alignment_fixed"] = True     # 이중 적용 방지 표시
    info["total_frames"] = running
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    (dst / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in out_eps))

    print(f"에피소드 {len(out_eps)} · 프레임 {running:,} (원본 {info['total_frames']+len(out_eps):,})")
    print(f"-> {dst}")
    print("ALIGN-FIX-DONE")


if __name__ == "__main__":
    main()
