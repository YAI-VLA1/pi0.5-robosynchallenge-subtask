import pathlib
p = pathlib.Path("/workspace/rsc_ws/RoboSynChallenge/policy/pi05/src/openpi/models/model.py")
s = p.read_text()
A = '                height, width = image.shape[1:3]'
N = ('                # MEM 이면 (B,T,H,W,C) 라 shape[1:3] 은 (T,H) 다. 그대로 쓰면\n'
     '                # RandomCrop 이 세로 T*0.95 픽셀로 잘라 이미지가 뭉개진다\n'
     '                # (cam_high 에만, 예외 없이 조용히).\n'
     '                height, width = image.shape[-3:-1]')
if "shape[-3:-1]\n" in s and "RandomCrop 이 세로" in s:
    print("이미 적용됨")
else:
    assert s.count(A) == 1, f"앵커 {s.count(A)}개"
    p.write_text(s.replace(A, N, 1)); print("OK — RandomCrop 치수를 뒤에서 세도록")
