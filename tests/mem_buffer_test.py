#!/usr/bin/env python3
"""추론 프레임 버퍼가 명세 §8 을 지키는지 본다 (모델 없이 로직만).

  · 정책 호출이 10 step 마다여도 25 step 간격 샘플링이 되는가
  · 에피소드 시작부에서 패딩 + invalid 인가
  · 미래 프레임을 쓰지 않는가
  · reset 후 히스토리가 에피소드를 넘지 않는가
"""
import os, sys
os.environ.setdefault("PI05_MEM_FRAMES", "6")
os.environ.setdefault("PI05_MEM_STRIDE_S", "1.0")
R = "/workspace/rsc_ws/RoboSynChallenge"
sys.path.insert(0, R)                      # policy.inference_timing 용
sys.path.insert(0, R + "/policy/pi05")
import numpy as np
import deploy_policy as D

print(f"T={D._MEM_T} · stride {D._MEM_STRIDE}s · fps {D._MEM_FPS} -> gap {D._MEM_GAP} step\n")

def frame(step):            # 스텝 번호를 픽셀값으로 심어 추적한다
    return [np.full((4, 4, 3), step, dtype=np.uint8) for _ in range(3)]

D.mem_reset()
ok = {}
for step in range(0, 140):
    D.mem_push(frame(step))
    if step in (0, 30, 130):
        hist, valid = D.mem_history()
        ids = [int(hist[0][k, 0, 0, 0]) for k in range(D._MEM_T)]
        print(f"step {step:3}  고른 스텝 {ids}  valid {valid[0].astype(int).tolist()}")
        if step == 0:
            ok["시작: 현재만 valid"] = valid[0].tolist() == [False]*5 + [True]
            ok["시작: 패딩은 현재 프레임"] = ids == [0]*6
        if step == 130:
            ok["간격 25 step"] = ids == [5, 30, 55, 80, 105, 130]
            ok["전부 valid"] = valid[0].all()
            ok["미래 없음"] = max(ids) <= step
        if step == 30:
            ok["중간: 없는 과거만 invalid"] = (valid[0].tolist() == [False]*4 + [True, True]
                                    and ids == [30, 30, 30, 30, 5, 30])

D.mem_reset()
D.mem_push(frame(77))
hist, valid = D.mem_history()
ids = [int(hist[0][k, 0, 0, 0]) for k in range(D._MEM_T)]
print(f"\nreset 후 첫 프레임  고른 스텝 {ids}  valid {valid[0].astype(int).tolist()}")
ok["reset 이 에피소드를 끊는다"] = ids == [77]*6 and valid[0].tolist() == [False]*5 + [True]
ok["shape"] = hist[0].shape == (6, 4, 4, 3) and len(hist) == 3

print()
for k, v in ok.items():
    print(f"  {'OK  ' if v else '✗   '}{k}")
print("\nMEM-BUFFER " + ("ALL-OK" if all(ok.values()) else "FAIL"))
if not all(ok.values()):
    raise SystemExit(1)   # 자동 게이트용 종료 코드 (리뷰 지적)
