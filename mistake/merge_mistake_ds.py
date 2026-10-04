#!/usr/bin/env python3
"""데모 1,000 + 실패 롤아웃 98 을 한 LeRobot v2.1 데이터셋으로 합친다.

설계
  · 데모 parquet / 영상, 롤아웃 영상은 **심볼릭 링크** 다. 복사가 없다.
  · 롤아웃 parquet 만 새로 쓴다 — [0, t_end] 로 자르고 episode/frame/index 를 다시 매긴다.
  · 영상은 자르지 않는다. LeRobot 은 parquet 길이만큼만 프레임을 요청하므로
    뒤에 남은 프레임은 읽히지 않는다 (AV1 재인코딩은 libaom 이라 몇 시간 걸린다).
  · task 문자열은 데모와 같은 하나를 쓴다. 같은 과제를 정책이 실행한 것이고,
    prompt_from_task 가 같은 프롬프트를 내야 한다.
  · meta/stats.json 은 데모 것을 그대로 쓴다. 롤아웃이 4% 라 통계는 사실상 같고,
    81999 체크포인트가 본 입력 분포를 바꾸지 않는 편이 재개에 안전하다.

mistake 시작 프레임은 meta/mistake_starts.json 에 {새 episode_index: t_mistake} 로 남긴다.
"""
from __future__ import annotations
import json, os, pathlib, shutil, sys

import numpy as np
import pandas as pd

# 경로는 환경변수로 받는다 (VESSL 기본값을 그대로 둔다).
#   PI05      = <repo>/policy/pi05
#   ROLLOUTS  = 실패 롤아웃 LeRobot 데이터셋 (mistake_labels.json 이 그 안에 있어야 한다)
_PI05 = pathlib.Path(os.environ.get(
    "PI05", "/workspace/rsc_ws/RoboSynChallenge/policy/pi05"))
TD   = _PI05 / "training_data/RoboSynChallenge"
DEMO = TD / "cobotmagic_Sim_items_handover"
ROLL = pathlib.Path(os.environ.get("ROLLOUTS", "/workspace/rollout_ds/subtask"))
OUT  = TD / "cobotmagic_Sim_items_handover_mix"
CAMS = ["observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]


def link(src: pathlib.Path, dst: pathlib.Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "meta").mkdir(parents=True)

    demo_info = json.loads((DEMO / "meta/info.json").read_text())
    demo_eps  = [json.loads(l) for l in (DEMO / "meta/episodes.jsonl").read_text().splitlines()]
    demo_task = json.loads((DEMO / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    n_demo    = demo_info["total_episodes"]
    assert n_demo == len(demo_eps) == 1000, (n_demo, len(demo_eps))

    labels = {r["ep"]: r for r in json.loads((ROLL / "mistake_labels.json").read_text())}
    use    = [r for r in labels.values() if r["use"]]
    use.sort(key=lambda r: r["ep"])
    print(f"데모 {n_demo} + 롤아웃 {len(use)}")

    # LeRobot 은 chunk = episode_index // chunks_size 로 경로를 만든다.
    # chunks_size 가 1000 이라 새 에피소드 1000~ 은 chunk-001 로 가야 한다.
    CH = demo_info["chunks_size"]

    def pq_path(root, ep, chunk=None):
        return root / f"data/chunk-{(ep // CH if chunk is None else chunk):03d}/episode_{ep:06d}.parquet"

    def vid_path(root, cam, ep, chunk=None):
        return root / f"videos/chunk-{(ep // CH if chunk is None else chunk):03d}/{cam}/episode_{ep:06d}.mp4"

    # ── 데모: parquet / 영상 링크 ─────────────────────────────────────────
    for e in range(n_demo):
        link(pq_path(DEMO, e), pq_path(OUT, e))
        for c in CAMS:
            link(vid_path(DEMO, c, e), vid_path(OUT, c, e))

    # ── 롤아웃: parquet 자르고 다시 매기기, 영상은 링크 ───────────────────
    episodes = list(demo_eps)
    running  = demo_info["total_frames"]
    starts   = {}
    for i, r in enumerate(use):
        new_ep = n_demo + i
        src = pq_path(ROLL, r["ep"], chunk=0)
        df  = pd.read_parquet(src).iloc[: r["n_keep"]].copy()
        assert len(df) == r["n_keep"], (len(df), r["n_keep"])
        df["episode_index"] = np.int64(new_ep)
        df["frame_index"]   = np.arange(len(df), dtype=np.int64)
        df["index"]         = np.arange(running, running + len(df), dtype=np.int64)
        df["task_index"]    = np.int64(0)
        df["timestamp"]     = (np.arange(len(df)) / demo_info["fps"]).astype(np.float32)
        dst = pq_path(OUT, new_ep)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        running += len(df)
        episodes.append({"episode_index": new_ep, "tasks": [demo_task], "length": len(df)})
        if r["t_mistake"] is not None:
            starts[new_ep] = r["t_mistake"]
        for c in CAMS:
            link(vid_path(ROLL, c, r["ep"], chunk=0), vid_path(OUT, c, new_ep))

    # ── meta ─────────────────────────────────────────────────────────────
    (OUT / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": demo_task}) + "\n")
    (OUT / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes))

    # episodes_stats: 데모 것 + 롤아웃 것(episode_index 만 다시 매김).
    # 롤아웃 통계는 자르기 전 전체 구간 기준이다 — openpi 는 자체 norm stats 를
    # 따로 계산해 쓰므로 학습에 들어가지 않는다.
    # pen_pose / holder_pose 는 두 데이터셋의 통계 shape 가 다르다 (데모 (4,) vs 롤아웃 (4,4)
    # — 데모 쪽에서 중첩 object 배열이 눌렸다). LeRobot 의 aggregate_stats 가 stack 에서
    # 터진다. 학습은 이 두 키를 읽지 않으므로 통계에서만 뺀다 (parquet 컬럼은 그대로 둔다).
    DROP = ("pen_pose", "holder_pose")

    def strip(rec):
        rec["stats"] = {k: v for k, v in rec["stats"].items() if k not in DROP}
        return rec

    out_stats = [json.dumps(strip(json.loads(l)))
                 for l in (DEMO / "meta/episodes_stats.jsonl").read_text().splitlines()]
    roll_stats = {json.loads(l)["episode_index"]: json.loads(l)
                  for l in (ROLL / "meta/episodes_stats.jsonl").read_text().splitlines()}
    for i, r in enumerate(use):
        rec = strip(roll_stats[r["ep"]])
        rec["episode_index"] = n_demo + i
        out_stats.append(json.dumps(rec))
    (OUT / "meta/episodes_stats.jsonl").write_text("\n".join(out_stats) + "\n")

    info = dict(demo_info)
    info["total_episodes"] = len(episodes)
    info["total_frames"]   = running
    info["total_videos"]   = len(episodes) * len(CAMS)
    info["splits"]         = {"train": f"0:{len(episodes)}"}
    (OUT / "meta/info.json").write_text(json.dumps(info, indent=4))
    shutil.copy(DEMO / "meta/stats.json", OUT / "meta/stats.json")

    (OUT / "meta/mistake_starts.json").write_text(json.dumps(starts, indent=1))

    n_mis = sum(episodes[e]["length"] - t for e, t in starts.items())
    print(f"에피소드 {len(episodes)} · 프레임 {running:,}")
    print(f"mistake=true 에피소드 {len(starts)} · 프레임 {n_mis:,} = {n_mis/running:.1%}")
    print(f"-> {OUT}")
    print("MERGE-DONE")


if __name__ == "__main__":
    main()
