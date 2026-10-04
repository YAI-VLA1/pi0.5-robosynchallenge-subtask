#!/usr/bin/env python3
"""MEM 추론 경로의 두 구멍을 막는다. (리뷰 R8)

① pi_model 의 transpose 가 3차원 고정이다
   버퍼는 T>1 일 때 (T,H,W,C) 를 준다. np.transpose(x, (2,0,1)) 는
   "axes don't match array" 로 죽는다. 차원을 보고 (T,H,W,C)->(T,C,H,W) 로 바꾼다.

② 추론에서 넣은 frame_valid 가 EmbodiChainInputs 에서 사라진다
   이 transform 은 학습용 observation/*_is_pad 에서만 frame_valid 를 만든다.
   dict 를 새로 만들므로 추론이 직접 넣은 키는 복사되지 않는다. 그대로 두면
   에피소드 초반의 '현재 프레임 복제 패딩'을 모델이 **진짜 과거로** 읽는다.
"""
import pathlib, sys

MARK = "RSC_MEMINFER"

# ── ① pi_model.py ──────────────────────────────────────────────────────
PM = "pi_model.py"
PM_A = '''        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))'''
PM_B = f'''        # {MARK}: MEM 이면 (T,H,W,C) 가 들어온다. 3차원 고정 transpose 는 죽는다.
        def _chw(x):
            x = np.asarray(x)
            if x.ndim == 4:        # (T,H,W,C) -> (T,C,H,W)
                return np.transpose(x, (0, 3, 1, 2))
            return np.transpose(x, (2, 0, 1))      # (H,W,C) -> (C,H,W)

        img_front = _chw(img_front)
        img_right = _chw(img_right)
        img_left = _chw(img_left)'''

# ── ② libero_policy.py (EmbodiChainInputs 안에서만) ─────────────────────
LP = "src/openpi/policies/libero_policy.py"
LP_A = '''            }} if "observation/image_is_pad" in data else {}),'''
# f-string 이라 중괄호는 두 배로 쓴다. 앵커의 `}}` 는 안쪽 dict 와 바깥 dict 를
# 둘 다 닫는 자리다 — 한 개로 줄이면 괄호가 안 맞는다 (실제로 한 번 깨뜨렸다).
LP_B = f'''            }}}} if "observation/image_is_pad" in data else
               # {MARK}: 추론은 repack 이 안 돌아 *_is_pad 가 없다. pi_model 이
               # 직접 넣은 frame_valid 를 여기서 통과시켜야 살아 간다.
               ({{"frame_valid": data["frame_valid"]}} if "frame_valid" in data else {{}})),'''


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    for rel, a, b, scope in ((PM, PM_A, PM_B, None),
                             (LP, LP_A, LP_B, "class EmbodiChainInputs")):
        p = root / rel
        if not p.exists():
            print(f"  {rel}: 파일 없음 — 건너뜀 (학습만 할 때는 정상)")
            continue
        s = p.read_text()
        if MARK in s:
            print(f"  {rel}: 이미 적용됨")
            continue
        lo, hi = 0, len(s)
        if scope:
            lo = s.index(scope)
            nxt = s.find("\nclass ", lo + 1)
            hi = nxt if nxt != -1 else len(s)
        body = s[lo:hi]
        n = body.count(a)
        if n != 1:
            raise SystemExit(f"★ {rel}: 앵커가 {n} 번 (1 이어야 함)\n{a.splitlines()[0][:90]}")
        s = s[:lo] + body.replace(a, b, 1) + s[hi:]
        p.write_text(s)
        print(f"  {rel}: ✓ 적용")


if __name__ == "__main__":
    main()
