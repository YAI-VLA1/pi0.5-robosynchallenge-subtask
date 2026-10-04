#!/usr/bin/env python3
"""MEM 버퍼를 **실제 평가 호출 순서**로 돌려 본다. (리뷰 R9)

기존 mem_buffer_test.py 는 env step 마다 push 를 딱 한 번 해서 이 문제를 놓쳤다.
진짜 루프는 이렇다:

    사이클마다:  encode_obs(obs)          <- 추론 직전 1회
                 for 10번: env.step(); encode_obs(final_obs)

즉 10 env step 당 push 11회이고, 사이클 첫 push 는 직전 사이클 마지막 관측과 같다.
"""
import importlib.util, pathlib, sys
import numpy as np

PI05 = pathlib.Path("/workspace/rsc_ws/RoboSynChallenge/policy/pi05")


def load():
    """deploy_policy.py 의 버퍼 부분만 떼어 import 한다 (모델 로딩 없이)."""
    src = (PI05 / "deploy_policy.py").read_text()
    # 버퍼 블록만 떼어 낸다 (`_MEM_T = ...` 부터 `def encode_obs(` 직전까지).
    # 모델·시뮬레이터 import 를 지나면 이 테스트가 무거워지고 GPU 를 잡는다.
    i = src.index("_MEM_T = ")
    head = "import numpy as np\nimport os as _os\n" + src[i: src.index("def encode_obs(")]
    ns = {"__name__": "mem_mod", "__file__": str(PI05 / "deploy_policy.py")}
    exec(compile(head, "deploy_policy_head", "exec"), ns)
    return ns


def main():
    ns = load()
    for k in ("mem_reset", "mem_push", "mem_history", "mem_tick"):
        if k not in ns:
            raise SystemExit(f"★ {k} 가 없다 — apply_mem_steptime.py 를 적용할 것")
    T, GAP = ns["_MEM_T"], ns["_MEM_GAP"]
    print(f"T={T} · GAP={GAP} env step")

    mem_reset, mem_push, mem_history, mem_tick = (
        ns["mem_reset"], ns["mem_push"], ns["mem_history"], ns["mem_tick"])

    # 이미지 대신 '그 step 번호' 를 담은 1x1 배열을 넣어 어떤 프레임이 뽑혔는지 센다.
    def frame(step):
        return [np.full((1, 1, 3), step % 256, np.uint8) for _ in range(3)]

    mem_reset()
    PI0_STEP = 10
    env_step = 0
    mem_push(frame(env_step))                       # 에피소드 시작, 추론 직전
    picks_at = {}
    for cycle in range(20):
        # --- 추론 직전 (같은 step 으로 다시 들어온다) ---
        mem_push(frame(env_step))
        hist, valid = mem_history()
        picks_at[env_step] = ([int(x[0, 0, 0]) for x in hist[0]],
                              [bool(v) for v in valid[0]])
        # --- 10 step 실행 ---
        for _ in range(PI0_STEP):
            env_step += 1
            mem_tick()
            mem_push(frame(env_step))

    ok = {}
    for s in (0, 50, 130, 200):
        if s not in picks_at:
            continue
        got, gval = picks_at[s]
        # 목표 step 이 음수면 아직 그 과거가 없다. 코드는 **현재 프레임으로 패딩**하고
        # valid=False 로 표시한다 (내용은 마스크되므로 무엇이든 상관없다).
        exp, eval_ = [], []
        for k in range(T - 1, -1, -1):
            t = s - k * GAP
            if t < 0:
                exp.append(s); eval_.append(False)
            else:
                exp.append(t); eval_.append(True)
        eval_[-1] = True
        same = got == exp and gval == eval_
        ok[f"step {s}"] = same
        print(f"  step {s:3}  프레임 {got}  valid {[int(v) for v in gval]}")
        print(f"           기대   {exp}  valid {[int(v) for v in eval_]}  {'OK' if same else '✗'}")

    # 버퍼 항목 수 == env step 수 + 1 (0번 포함) 이어야 한다
    buf_steps = [s for s, _ in ns["_mem_buf"]]
    ok["버퍼에 중복 step 없음"] = len(buf_steps) == len(set(buf_steps))
    ok["버퍼 step 이 1씩 증가"] = all(b - a == 1 for a, b in zip(buf_steps, buf_steps[1:]))
    print(f"\n  버퍼 step 들 {buf_steps[:3]} ... {buf_steps[-3:]}")

    print()
    for k, v in ok.items():
        print(f"  {'OK ' if v else '✗  '} {k}")
    if not all(ok.values()):
        raise SystemExit("MEM-BUFFER-EVALORDER FAIL")
    print("\nMEM-BUFFER-EVALORDER OK")


if __name__ == "__main__":
    main()
