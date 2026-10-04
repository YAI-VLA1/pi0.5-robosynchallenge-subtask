import pathlib, sys
# 경로 계약: 인자 1개 = <repo>/policy/pi05 (PI05 루트). 생략하면 VESSL 기본값.
# 예전에는 이 값을 상수로 박아 둬서 다른 머신에서 apply_all.sh 가
# 엉뚱한 checkout 을 고치거나 FileNotFoundError 를 냈다. (리뷰 R7)
_PI05 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                     else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
p = _PI05 / "pi_model.py"
s = p.read_text()
A = "    def update_observation_window(self, img_arr, state):"
N = "    def update_observation_window(self, img_arr, state, frame_valid=None):"
A2 = '''            "observation/state": state,
            "prompt": self.instruction,
        }'''
N2 = '''            "observation/state": state,
            "prompt": self.instruction,
        }
        # MEM: 프레임별 유효 마스크. 키 이름은 EmbodiChainInputs 가 쓰는 것과 맞춘다.
        if frame_valid is not None:
            self.observation_window["frame_valid"] = {
                "base_0_rgb": frame_valid[0],
                "right_wrist_0_rgb": frame_valid[1],
                "left_wrist_0_rgb": frame_valid[2],
            }'''
if "frame_valid=None" in s:
    print("이미 적용됨")
else:
    for a in (A, A2):
        assert s.count(a) == 1, f"앵커 {s.count(a)}개"
    p.write_text(s.replace(A, N, 1).replace(A2, N2, 1))
    print("OK — update_observation_window 가 frame_valid 를 받는다")
