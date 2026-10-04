#!/usr/bin/env python3
"""MEM 버퍼의 시간축을 **실제 env step** 으로 바꾼다. (리뷰 R9)

무엇이 문제인가
  mem_push 가 호출될 때마다 자체 카운터를 1 올렸다. 그런데 평가 루프는
  추론 직전에 한 번, 그리고 실행 중 매 step 뒤에 한 번 encode_obs 를 부른다.
  10 env step 당 push 가 **11번** 일어나고, 사이클 첫 push 는 직전 사이클
  마지막 관측과 같은 것이다.

  결과: step 130 에서 뽑힌 히스토리가 [17,40,62,85,108,130] 이었다.
  25 env step 간격으로 기대한 [5,30,55,80,105,130] 와 다르다.

어떻게 고치나
  env.step 직후에만 올라가는 카운터(mem_tick)를 둔다. 같은 step 번호로
  다시 push 되면 쌓지 않고 **덮어쓴다**. 그러면 버퍼의 항목 수 = env step 수다.
"""
import pathlib, sys

MARK = "RSC_MEMSTEP"
F = "deploy_policy.py"

A = '''def mem_reset():
    """에피소드 경계에서 버퍼를 비운다. 안 비우면 히스토리가 에피소드를 넘어간다."""
    _mem_buf.clear()
    _mem_step[0] = 0


def mem_push(img_arr):
    """매 encode_obs 호출마다 현재 프레임을 쌓는다. 스텝 번호를 같이 남긴다."""
    _mem_buf.append((_mem_step[0], [np.asarray(x) for x in img_arr]))
    _mem_step[0] += 1
    keep = (_MEM_T - 1) * _MEM_GAP + 1
    while len(_mem_buf) > keep + 2:
        _mem_buf.pop(0)'''

B = f'''def mem_reset():
    """에피소드 경계에서 버퍼를 비운다. 안 비우면 히스토리가 에피소드를 넘어간다."""
    _mem_buf.clear()
    _mem_step[0] = 0


def mem_tick():
    """{MARK}: env.step 직후에만 부른다. 버퍼의 시간축은 이 값이다.

    encode_obs 호출 횟수로 세면 안 된다 — 평가 루프는 추론 직전에 한 번,
    실행 중 매 step 뒤에 한 번 불러서 10 step 당 11번이 된다.
    """
    _mem_step[0] += 1


def mem_push(img_arr):
    """현재 프레임을 쌓는다. 같은 env step 으로 다시 들어오면 덮어쓴다."""
    s = _mem_step[0]
    entry = (s, [np.asarray(x) for x in img_arr])
    if _mem_buf and _mem_buf[-1][0] == s:
        _mem_buf[-1] = entry          # {MARK}: 같은 스텝 중복 push
    else:
        _mem_buf.append(entry)
    keep = (_MEM_T - 1) * _MEM_GAP + 1
    while len(_mem_buf) > keep + 2:
        _mem_buf.pop(0)'''

A2 = '''        final_obs, reward, terminated, truncated, info = env.step(action_tensor)'''
B2 = f'''        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        mem_tick()          # {MARK}: 버퍼 시간축은 env step 수다'''


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    p = root / F
    if not p.exists():
        print(f"  {F}: 파일 없음 — 건너뜀")
        return
    s = p.read_text()
    if MARK in s:
        print(f"  {F}: 이미 적용됨")
        return
    for a in (A, A2):
        n = s.count(a)
        if n != 1:
            raise SystemExit(f"★ {F}: 앵커가 {n} 번 (1 이어야 함)\n{a.splitlines()[0][:90]}")
    p.write_text(s.replace(A, B, 1).replace(A2, B2, 1))
    print(f"  {F}: ✓ 적용")


if __name__ == "__main__":
    main()
