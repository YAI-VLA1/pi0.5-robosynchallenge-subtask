import pathlib, sys
# 경로 계약: 인자 1개 = <repo>/policy/pi05 (PI05 루트). 생략하면 VESSL 기본값.
# 예전에는 이 값을 상수로 박아 둬서 다른 머신에서 apply_all.sh 가
# 엉뚱한 checkout 을 고치거나 FileNotFoundError 를 냈다. (리뷰 R7)
_PI05 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                     else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
P = _PI05

# ① config.py 의 items_handover repack — _is_pad 를 통과시킨다
c = P / "src/openpi/training/config.py"
s = c.read_text()
A = '                        "observation/state": self.state_key,\n                        "actions": "action",'
N = ('                        "observation/state": self.state_key,\n'
     '                        "actions": "action",\n'
     '                        # MEM: repack 은 매핑에 없는 키를 버린다. LeRobot 의\n'
     '                        # 프레임 유효 마스크를 여기서 명시해야 frame_valid 가 살아 간다.\n'
     '                        **({"observation/image_is_pad": self.image_key_high + "_is_pad",\n'
     '                            "observation/left_wrist_image_is_pad": self.image_key_left + "_is_pad",\n'
     '                            "observation/right_wrist_image_is_pad": self.image_key_right + "_is_pad"}\n'
     '                           if _os.environ.get("PI05_MEM_FRAMES", "1") != "1" else {}),')
if "PI05_MEM_FRAMES" in s:
    print("  config.py: 이미 적용됨")
else:
    assert s.count(A) == 1, f"config.py 앵커 {s.count(A)}개"
    s = s.replace(A, N, 1)
    if "\nimport os as _os\n" not in s:
        s = s.replace("import dataclasses\n", "import dataclasses\nimport os as _os\n", 1)
    c.write_text(s); print("  config.py: 적용")

# ② libero_policy.py 의 EmbodiChainInputs 에만 frame_valid 를 넣는다 (클래스 범위 지정)
l = P / "src/openpi/policies/libero_policy.py"
t = l.read_text()
if "frame_valid" in t:
    print("  libero_policy.py: 이미 적용됨"); sys.exit(0)
head = "class EmbodiChainInputs(transforms.DataTransformFn):"
i = t.index(head)
body, rest = t[i:], t[:i]
A2 = '            "image_mask": {'
assert body.count(A2) == 1, f"EmbodiChainInputs 안 앵커 {body.count(A2)}개"
N2 = ('            # MEM: is_pad(True=패딩) 를 뒤집어 frame_valid 로 만든다.\n'
      '            **({"frame_valid": {\n'
      '                "base_0_rgb": ~np.asarray(data["observation/image_is_pad"]),\n'
      '                "left_wrist_0_rgb": ~np.asarray(data["observation/left_wrist_image_is_pad"]),\n'
      '                "right_wrist_0_rgb": ~np.asarray(data["observation/right_wrist_image_is_pad"]),\n'
      '            }} if "observation/image_is_pad" in data else {}),\n'
      '            "image_mask": {')
l.write_text(rest + body.replace(A2, N2, 1)); print("  libero_policy.py: 적용 (EmbodiChainInputs 범위)")
