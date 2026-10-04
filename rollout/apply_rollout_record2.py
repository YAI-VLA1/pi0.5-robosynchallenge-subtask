#!/usr/bin/env python3
"""eval_policy 가 롤아웃을 LeRobot v2.1 로 저장하도록 한다 (RolloutRecorder 사용).

앞선 시도(내장 LeRobotRecorder)는 lerobot 버전 충돌로 실패했다 —
EmbodiChain 은 신 API, openpi 는 구 API. 그래서 구 API 로 직접 쓴다.

PI05_RECORD_ROLLOUT=<저장경로> 일 때만 켜진다.
"""
import pathlib, sys

F = "scripts/eval_policy.py"

# 0) 앞선 패치 되돌리기
OLD1 = '''    env_cfg.filter_dataset_saving = bool(config.get("filter_dataset_saving", True))
    # PI05_RECORD_ROLLOUT=1 이면 성공 필터를 끈다 — 실패 에피소드도 저장해야 한다.
    import os as _os
    if _os.environ.get("PI05_RECORD_ROLLOUT") == "1":
        env_cfg.filter_dataset_saving = False
        print("[record] filter_dataset_saving=False (실패도 저장)", flush=True)'''
NEW1 = '''    env_cfg.filter_dataset_saving = bool(config.get("filter_dataset_saving", True))'''

OLD2 = '''                _rec = __import__("os").environ.get("PI05_RECORD_ROLLOUT") == "1"
                # save_data=True 면 리셋 시점에 **직전** 에피소드가 저장된다.
                # 첫 리셋에는 저장할 게 없으므로 그대로 둔다.
                if _rec:
                    obs, info = eval_env.reset(seed=ep_seed,
                                               options={"save_data": episode > 0})
                else:
                    obs, info = eval_env.reset(seed=ep_seed)'''
NEW2 = '''                obs, info = eval_env.reset(seed=ep_seed)'''

# 1) 프록시가 (obs, action) 을 넘기도록
A_STEP = '''    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._recorder and not (
            self._as_done(terminated) or self._as_done(truncated)
        ):
            self._recorder.record(obs)
        return obs, reward, terminated, truncated, info'''
N_STEP = '''    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = self._as_done(terminated) or self._as_done(truncated)
        if self._recorder and not done:
            self._recorder.record(obs)
        # 롤아웃 기록 — (obs, action) 쌍이 여기서만 같이 보인다
        if getattr(self, "_roll", None) is not None and not done:
            self._roll.record(obs, action)
        return obs, reward, terminated, truncated, info'''

# 2) 생성
A_MK = "    video_recorder = create_video_recorder(config, run_dir=run_dir)"
N_MK = '''    video_recorder = create_video_recorder(config, run_dir=run_dir)
    # ── 롤아웃 기록 (PI05_RECORD_ROLLOUT=<경로>) ─────────────────────────
    import os as _os
    _roll_rec = None
    _roll_dir = _os.environ.get("PI05_RECORD_ROLLOUT")
    if _roll_dir:
        sys.path.insert(0, "/root/rsc_recover")
        from rollout_recorder import RolloutRecorder
        _roll_rec = RolloutRecorder(
            root=_roll_dir,
            repo_id=_os.environ.get("PI05_RECORD_REPO", "yai-robosync/rollout"),
            task=str(config.get("instruction") or "rollout"))'''

# 3) 프록시에 달기
A_ATT = "            if video_recorder:\n                video_recorder.start_episode(episode, ep_seed)"
N_ATT = '''            if video_recorder:
                video_recorder.start_episode(episode, ep_seed)
            if _roll_rec is not None:
                eval_env._roll = _roll_rec'''

# 4) 에피소드 종료 / 전체 종료
A_END = '''                if video_recorder:
                    video_recorder.close_episode(success=episode_success)'''
N_END = '''                if video_recorder:
                    video_recorder.close_episode(success=episode_success)
                if _roll_rec is not None:
                    _roll_rec.end_episode(episode_success, ep_seed,
                                          _os.environ.get("PI05_RECORD_SRC", "unknown"))'''

A_FIN = '''    finally:
        if video_recorder:
            video_recorder.close_episode(success=False)'''
N_FIN = '''    finally:
        if video_recorder:
            video_recorder.close_episode(success=False)
        if _roll_rec is not None:
            _roll_rec.finish()'''


def main():
    p = pathlib.Path(sys.argv[1]) / F
    s = p.read_text()
    if "RolloutRecorder" in s:
        print("이미 적용됨"); return
    for a, n in ((OLD1, NEW1), (OLD2, NEW2)):
        if a in s:
            s = s.replace(a, n, 1); print("  이전 패치 되돌림")
    for a, n in ((A_STEP, N_STEP), (A_MK, N_MK), (A_ATT, N_ATT), (A_END, N_END), (A_FIN, N_FIN)):
        if s.count(a) != 1:
            sys.exit(f"앵커가 {s.count(a)}개 — 중단:\n{a[:90]}")
        s = s.replace(a, n, 1)
    p.write_text(s); print(f"패치 완료 {p}")


if __name__ == "__main__":
    main()
