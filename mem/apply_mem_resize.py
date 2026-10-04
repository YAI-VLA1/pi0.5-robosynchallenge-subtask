import pathlib, sys
# 경로 계약: 인자 1개 = <repo>/policy/pi05 (PI05 루트). 생략하면 VESSL 기본값.
# 예전에는 이 값을 상수로 박아 둬서 다른 머신에서 apply_all.sh 가
# 엉뚱한 checkout 을 고치거나 FileNotFoundError 를 냈다. (리뷰 R7)
_PI05 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                     else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
p = _PI05 / "src/openpi/shared/image_tools.py"
s = p.read_text()
A = """    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # type: ignore"""
N = """    # MEM 으로 (B,T,H,W,C) 처럼 앞 차원이 둘 이상일 수 있다. ndim==4 만 배치로
    # 보면 5차원 입력에 차원을 하나 더 붙여 jax.image.resize 가 터진다.
    # 앞쪽 차원을 전부 접었다가 끝에 되돌린다 (프레임마다 독립 리사이즈가 맞다).
    lead = images.shape[:-3]
    if len(lead) != 1:
        images = images.reshape(-1, *images.shape[-3:])  # type: ignore
    has_batch_dim = True"""
A2 = """    if not has_batch_dim:
        padded_images = padded_images[0]
    return padded_images"""
N2 = """    if len(lead) != 1:
        padded_images = padded_images.reshape(*lead, *padded_images.shape[-3:])
    return padded_images"""
if "MEM 으로 (B,T,H,W,C)" in s:
    print("이미 적용됨")
else:
    for a in (A, A2):
        assert s.count(a) == 1, f"앵커 {s.count(a)}개"
    p.write_text(s.replace(A, N, 1).replace(A2, N2, 1))
    print("OK — resize_with_pad 를 임의 선행 차원으로 일반화")
