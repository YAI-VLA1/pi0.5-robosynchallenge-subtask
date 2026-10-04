#!/usr/bin/env python3
"""평가 롤아웃을 LeRobot v2.1 로 저장한다 (구 lerobot 0.1.0 API 사용).

왜 직접 쓰나
  EmbodiChain 의 LeRobotRecorder 는 신 API(lerobot.datasets)를 요구하는데
  openpi 는 구 API(lerobot.common.datasets)에 묶여 있어 한 venv 에 공존 못 한다.
  구 API 에도 LeRobotDataset.create 가 있으므로 직접 쓴다.

스키마는 기존 학습 데이터와 **동일**하게 맞춘다 — 그래야 그대로 섞어 쓸 수 있다.
mistake 같은 메타는 스키마를 건드리지 않고 사이드카 JSON 으로 뺀다.
"""
from __future__ import annotations
import json, os, pathlib
import numpy as np

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
    "observation.qvel":  {"dtype": "float32", "shape": (14,), "names": None},
    "observation.qf":    {"dtype": "float32", "shape": (14,), "names": None},
    "action":            {"dtype": "float32", "shape": (14,), "names": None},
    "observation.images.cam_high":        {"dtype": "video", "shape": (480,640,3), "names": ["height","width","channel"]},
    "observation.images.cam_right_wrist": {"dtype": "video", "shape": (480,640,3), "names": ["height","width","channel"]},
    "observation.images.cam_left_wrist":  {"dtype": "video", "shape": (480,640,3), "names": ["height","width","channel"]},
    "pen_pose":    {"dtype": "float32", "shape": (4,4), "names": None},
    "holder_pose": {"dtype": "float32", "shape": (4,4), "names": None},
}
CAMS = [("observation.images.cam_high", "cam_high"),
        ("observation.images.cam_right_wrist", "cam_right_wrist"),
        ("observation.images.cam_left_wrist", "cam_left_wrist")]


def _np(x):
    if x is None: return None
    if hasattr(x, "detach"): x = x.detach().cpu()
    return np.asarray(x)


def _first(obs, *keys):
    """중첩 dict 에서 첫 번째로 찾히는 키를 꺼낸다 (구조가 버전마다 다르다)."""
    for k in keys:
        cur, ok = obs, True
        for part in k.split("/"):
            try:                      # dict 와 TensorDict 를 모두 받는다
                cur = cur[part]
            except Exception:
                ok = False; break
        if ok: return cur
    return None


class RolloutRecorder:
    def __init__(self, root: str, repo_id: str, task: str, fps: int = 25):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        self.root = pathlib.Path(root)
        self.task = task
        if (self.root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset(repo_id, root=self.root)
            print(f"[rec] 기존 데이터셋에 이어쓰기 ({self.ds.num_episodes} 에피소드)", flush=True)
        else:
            self.ds = LeRobotDataset.create(
                repo_id=repo_id, fps=fps, root=self.root,
                robot_type="cobotmagic", features=FEATURES,
                use_videos=True, image_writer_threads=4)
            print(f"[rec] 새 데이터셋 생성 {self.root}", flush=True)
        # 재시작 수집: 기존 사이드카를 읽고 번호를 데이터셋에 맞춘다.
        # 빈 리스트로 시작하면 다음 end_episode 에서 rollout_meta.json 을
        # 새 기록만으로 **덮어쓰고** episode 번호도 0 부터 다시 매긴다.
        # 그러면 parquet 의 episode_index 와 seed/success 가 어긋나 이후
        # mistake 라벨링이 엉뚱한 에피소드에 붙는다. (리뷰 R13)
        side = self.root / "rollout_meta.json"
        self.meta = json.loads(side.read_text()) if side.exists() else []
        self._n = int(self.ds.num_episodes)
        if self.meta and self.meta[-1]["episode"] + 1 != self._n:
            print(f"[rec] ★ 사이드카({self.meta[-1]['episode'] + 1})와 "
                  f"데이터셋({self._n}) 에피소드 수가 다르다 — 사이드카를 데이터셋에 맞춘다",
                  flush=True)
            self.meta = [m for m in self.meta if m["episode"] < self._n]
        self._cur = []
        self._probed = False

    def record(self, obs, action):
        st  = _np(_first(obs, "robot/qpos"))
        qv  = _np(_first(obs, "robot/qvel"))
        qf  = _np(_first(obs, "robot/qf"))
        pen = _np(_first(obs, "pen_pose", "object/pen_pose"))
        hol = _np(_first(obs, "holder_pose", "object/holder_pose"))
        if not self._probed:
            self._probed = True
            print(f"[rec] obs 최상위 키: {sorted(map(str, obs.keys()))}", flush=True)
            for n, v in (("qpos",st),("qvel",qv),("qf",qf),("pen",pen),("holder",hol)):
                print(f"[rec]   {n:7} {None if v is None else v.shape}", flush=True)
        def row(v, d):
            if v is None: return np.zeros(d, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            return (v[:np.prod(d)] if v.size >= np.prod(d)
                    else np.pad(v, (0, int(np.prod(d))-v.size))).reshape(d)
        f = {
            "observation.state": row(st, (14,)),
            "observation.qvel":  row(qv, (14,)),
            "observation.qf":    row(qf, (14,)),
            "action":            row(_np(action), (14,)),
            "pen_pose":          row(pen, (4,4)),
            "holder_pose":       row(hol, (4,4)),
        }
        for feat, cam in CAMS:
            img = _np(_first(obs, f"sensor/{cam}/color"))
            if img is None:
                img = np.zeros((480,640,3), dtype=np.uint8)
            else:
                img = img[0] if img.ndim == 4 else img
                img = img[..., :3].astype(np.uint8)
            f[feat] = img
        f["task"] = self.task
        self.ds.add_frame(f)
        self._cur.append(1)

    def end_episode(self, success: bool, seed: int, source: str):
        if not self._cur:
            return
        n = len(self._cur)
        self.ds.save_episode()
        self.meta.append({"episode": self._n, "length": n,
                          "success": bool(success), "seed": int(seed),
                          "source": source, "mistake": (not success)})
        # atomic write — 수집 중 파드가 죽어도 반쪽짜리 JSON 이 남지 않는다.
        side = self.root / "rollout_meta.json"
        tmp = side.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.meta, indent=1))
        tmp.replace(side)
        print(f"[rec] ep{self._n} 저장 ({n} 프레임, {'성공' if success else '실패'})", flush=True)
        self._n += 1
        self._cur = []

    def finish(self):
        try:
            self.ds.consolidate()
        except Exception as e:
            print(f"[rec] consolidate 생략: {type(e).__name__}", flush=True)
        print(f"[rec] 완료 — {self._n} 에피소드 -> {self.root}", flush=True)
