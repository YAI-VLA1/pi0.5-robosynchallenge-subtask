#!/usr/bin/env python3
"""롤아웃 녹화의 (obs, action) 정렬을 고친다. (리뷰 R1)

무엇이 문제였나
  step() 이 반환한 **실행 후** obs 와 방금 실행한 action 을 같이 저장했다.
  BC 가 필요한 (o_t, a_t) 대신 (o_{t+1}, a_t) 가 들어간다.

  수치로 확인한 증거 (팔 12축 |action - state| 중앙값):
      대본 데모    |a[t]-s[t]| 0.0058   |a[t]-s[t+1]| 0.0183   <- a[t] 가 s[t] 와 짝
      롤아웃      |a[t]-s[t]| 0.0517   |a[t]-s[t+1]| 0.0354   <- 한 칸 어긋남
  action 은 절대 qpos 명령이라 같은 시점끼리 가장 가깝다.

어떻게 고치나
  reset 직후와 매 step 직전의 관측을 _roll_prev 에 들고 있다가, step 에서
  **그 관측**과 지금 넣는 action 을 짝지어 기록한다. 환경이 obs 버퍼를
  재사용할 수 있으므로 보관 시 복사한다.

  done 을 낸 마지막 action 도 pre-step 관측과 함께 남긴다 — 그 쌍은 정상이고,
  버리면 에피소드 끝(실패 직후)이 사라진다.
"""
import pathlib, sys

MARK = "RSC_ALIGN"
F = "scripts/eval_policy.py"

A_RESET = '''    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs, info = self._sync_reset_obs(obs, info)
        if self._recorder:
            self._recorder.record(obs)
        return obs, info'''
B_RESET = f'''    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs, info = self._sync_reset_obs(obs, info)
        if self._recorder:
            self._recorder.record(obs)
        # {MARK}: 정책이 다음 action 을 만들 때 보는 관측. step 에서 이것과 짝짓는다.
        self._roll_prev = self._roll_copy(obs)
        return obs, info

    @staticmethod
    def _roll_copy(obs):
        """환경이 obs 버퍼를 재사용할 수 있다. 보관 전에 복사한다."""
        import copy as _copy
        try:
            return _copy.deepcopy(obs)
        except Exception:
            return obs'''

A_STEP = '''    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = self._as_done(terminated) or self._as_done(truncated)
        if self._recorder and not done:
            self._recorder.record(obs)
        # 롤아웃 기록 — (obs, action) 쌍이 여기서만 같이 보인다
        if getattr(self, "_roll", None) is not None and not done:
            self._roll.record(obs, action)
        return obs, reward, terminated, truncated, info'''
B_STEP = f'''    def step(self, action):
        # {MARK}: 이 action 을 만들어 낸 관측은 **step 이전** 것이다.
        prev = getattr(self, "_roll_prev", None)
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = self._as_done(terminated) or self._as_done(truncated)
        if self._recorder and not done:
            self._recorder.record(obs)
        # 롤아웃 기록 — (o_t, a_t). done 을 낸 마지막 쌍도 남긴다(그 쌍은 정상이다).
        if getattr(self, "_roll", None) is not None and prev is not None:
            self._roll.record(prev, action)
        self._roll_prev = self._roll_copy(obs)
        return obs, reward, terminated, truncated, info'''


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge")
    p = root / F
    s = p.read_text()
    if MARK in s:
        print(f"  {F}: 이미 적용됨")
        return
    for a in (A_RESET, A_STEP):
        n = s.count(a)
        if n != 1:
            raise SystemExit(f"★ {F}: 앵커가 {n} 번 (1 이어야 함)\n{a.splitlines()[0]}")
    s = s.replace(A_RESET, B_RESET, 1).replace(A_STEP, B_STEP, 1)
    p.write_text(s)
    print(f"  {F}: ✓ 적용")


if __name__ == "__main__":
    main()
