import pathlib, sys
# 경로 계약: 인자 1개 = <repo>/policy/pi05 (PI05 루트). 생략하면 VESSL 기본값.
# 예전에는 이 값을 상수로 박아 둬서 다른 머신에서 apply_all.sh 가
# 엉뚱한 checkout 을 고치거나 FileNotFoundError 를 냈다. (리뷰 R7)
_PI05 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                     else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
p = _PI05 / "src/openpi/models/model.py"
s = p.read_text()
A1 = """        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")"""
N1 = """        # MEM 이면 (B,T,H,W,C) 라 shape[1:3] 이 (T,H) 가 된다. 뒤에서 센다.
        if image.shape[-3:-1] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[-3:-1]} to {image_resolution}")"""
A2 = """            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)"""
N2 = """            sub_rngs = jax.random.split(rng, image.shape[0])
            _chain = augmax.Chain(*transforms)
            if image.ndim == 5:
                # MEM: 한 카메라의 모든 프레임에 **같은** 기하 변환을 건다.
                # 프레임마다 독립 crop/rotate 를 하면 같은 patch 위치의
                # 시간 대응이 깨져 시간 어텐션이 엉뚱한 패치를 잇는다.
                image = jax.vmap(
                    lambda r, ims: jax.vmap(_chain, in_axes=(None, 0))(r, ims)
                )(sub_rngs, image)
            else:
                image = jax.vmap(_chain)(sub_rngs, image)"""
if "MEM: 한 카메라의 모든 프레임에" in s:
    print("이미 적용됨")
else:
    for a in (A1, A2):
        assert s.count(a) == 1, f"앵커 {s.count(a)}개"
    p.write_text(s.replace(A1, N1, 1).replace(A2, N2, 1))
    print("OK — 증강을 프레임 공통으로, 해상도 검사를 뒤에서 세도록")
