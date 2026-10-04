#!/usr/bin/env python3
"""추론 때 과거 프레임 버퍼를 둔다 (MEM).

왜 스텝 인덱스가 필요한가
  정책 호출은 10 env step 마다지만 encode_obs 는 매 step 불린다
  (추론 직전 1 + 실행 중 10 = 사이클당 11). 호출마다 한 장씩 쌓고 stride 25 를
  쓰면 1초 간격이 안 된다 — 들어온 순서가 아니라 **스텝 번호**로 골라야 한다.

규칙 (명세 §8)
  · 과거만 쓴다. 미래 프레임 금지
  · 에피소드 시작 이전은 패딩 + invalid 표시. 다른 에피소드 프레임을 끌어오지 않는다
  · 현재 프레임은 마지막 칸이며 항상 valid
  · reset_model 에서 버퍼를 비운다 (안 비우면 히스토리가 에피소드를 넘어간다)

PI05_MEM_FRAMES (기본 1 = 끔) · PI05_MEM_STRIDE_S (기본 1.0) · 25 fps 가정
"""
import pathlib, sys

# 경로 계약을 다른 MEM 패치와 맞춘다: 인자 = PI05 루트 (리뷰 R7)
F = "deploy_policy.py"

A_HEAD = "def encode_obs(obs):"
N_HEAD = '''# ── MEM 추론 프레임 버퍼 ────────────────────────────────────────────
import os as _os

_MEM_T = int(_os.environ.get("PI05_MEM_FRAMES", "1"))
_MEM_STRIDE = float(_os.environ.get("PI05_MEM_STRIDE_S", "1.0"))
_MEM_FPS = float(_os.environ.get("PI05_MEM_FPS", "25"))
_MEM_GAP = max(1, int(round(_MEM_STRIDE * _MEM_FPS)))   # 1초 = 25 step

_mem_buf = []      # [(step, [img_front, img_right, img_left])]
_mem_step = [0]


def mem_reset():
    """에피소드 경계에서 버퍼를 비운다. 안 비우면 히스토리가 에피소드를 넘어간다."""
    _mem_buf.clear()
    _mem_step[0] = 0


def mem_push(img_arr):
    """매 encode_obs 호출마다 현재 프레임을 쌓는다. 스텝 번호를 같이 남긴다."""
    _mem_buf.append((_mem_step[0], [np.asarray(x) for x in img_arr]))
    _mem_step[0] += 1
    keep = (_MEM_T - 1) * _MEM_GAP + 1
    while len(_mem_buf) > keep + 2:
        _mem_buf.pop(0)


def mem_history():
    """(히스토리 3뷰, frame_valid 3뷰) 를 돌려준다. 현재가 마지막 칸이다.

    목표 스텝보다 **이전** 중 가장 가까운 프레임을 쓴다 (미래 금지).
    에피소드 시작 전이면 현재 프레임으로 패딩하고 valid=False 로 표시한다.
    """
    cur_step, cur_imgs = _mem_buf[-1]
    n_views = len(cur_imgs)
    hist = [[] for _ in range(n_views)]
    valid = []
    for k in range(_MEM_T - 1, -1, -1):          # oldest -> current
        target = cur_step - k * _MEM_GAP
        pick, ok = None, False
        if target >= 0:
            for st, imgs in reversed(_mem_buf):   # target 이하 중 가장 가까운 것
                if st <= target:
                    pick, ok = imgs, True
                    break
        if pick is None:
            pick = cur_imgs                       # 패딩 (invalid 로 표시된다)
        for v in range(n_views):
            hist[v].append(pick[v])
        valid.append(ok)
    valid[-1] = True                              # 현재는 항상 valid
    return ([np.stack(h, axis=0) for h in hist],
            [np.asarray(valid, dtype=bool) for _ in range(n_views)])


def encode_obs(obs):'''

A_RET = """    img_arr = [img_front, img_right, img_left]

    return img_arr, state"""
N_RET = """    img_arr = [img_front, img_right, img_left]

    if _MEM_T > 1:
        mem_push(img_arr)
        img_arr, _fv = mem_history()
        return img_arr, state, _fv
    return img_arr, state"""

A_RESET = '''def reset_model(model):
    """Reset π₀ internal state (observation window and instruction)."""
    model.reset_obsrvationwindows()'''
N_RESET = '''def reset_model(model):
    """Reset π₀ internal state (observation window and instruction)."""
    model.reset_obsrvationwindows()
    mem_reset()'''

# eval() 안의 두 호출부가 3-튜플도 받게 한다
A_E1 = """    img_arr, state = encode_obs(obs)
    model.update_observation_window(img_arr, state)"""
N_E1 = """    _enc = encode_obs(obs)
    model.update_observation_window(*_enc)"""
A_E2 = """        img_arr, state = encode_obs(final_obs)
        model.update_observation_window(img_arr, state)"""
N_E2 = """        _enc = encode_obs(final_obs)
        model.update_observation_window(*_enc)"""


def main():
    p = pathlib.Path(sys.argv[1]) / F
    s = p.read_text()
    if "_mem_buf" in s:
        print("이미 적용됨"); return
    pairs = ((A_HEAD, N_HEAD), (A_RET, N_RET), (A_RESET, N_RESET), (A_E1, N_E1), (A_E2, N_E2))
    for a, _ in pairs:
        if s.count(a) != 1:
            sys.exit(f"앵커가 {s.count(a)}개 — 중단:\\n{a[:70]}")
    for a, n in pairs:
        s = s.replace(a, n, 1)
    if "import numpy as np" not in s:
        s = s.replace("import os as _os", "import numpy as np\nimport os as _os", 1)
    p.write_text(s); print(f"패치 완료 {p}")


if __name__ == "__main__":
    main()
