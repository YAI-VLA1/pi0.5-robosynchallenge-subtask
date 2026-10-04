import pathlib, sys
# 경로 계약: 인자 1개 = <repo>/policy/pi05 (PI05 루트). 생략하면 VESSL 기본값.
# 예전에는 이 값을 상수로 박아 둬서 다른 머신에서 apply_all.sh 가
# 엉뚱한 checkout 을 고치거나 FileNotFoundError 를 냈다. (리뷰 R7)
_PI05 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                     else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
P = _PI05

# ① pi0.py — 비전 모듈에 num_frames 를 준다
f = P / "src/openpi/models/pi0.py"
s = f.read_text()
A = '''            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )'''
N = '''            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
                # MEM: 데이터 로더와 같은 환경변수를 본다. 1 이면 기존과 동일하다.
                num_frames=int(__import__("os").environ.get("PI05_MEM_FRAMES", "1")),
            )'''
if "PI05_MEM_FRAMES" in s:
    print("  pi0.py: 이미 적용됨")
else:
    assert s.count(A) == 1, f"pi0.py 앵커 {s.count(A)}개"
    f.write_text(s.replace(A, N, 1)); print("  pi0.py: 적용")

# ② pi0_config.py — fake_obs 도 히스토리 모양이어야 lazy_init 이 통과한다
g = P / "src/openpi/models/pi0_config.py"
t = g.read_text()
A2 = '        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)\n        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)'
N2 = ('        # MEM: T>1 이면 (B,T,H,W,3). lazy_init 이 이 모양으로 초기화해야 한다.\n'
      '        _T = int(__import__("os").environ.get("PI05_MEM_FRAMES", "1"))\n'
      '        _shape = ([batch_size, *_model.IMAGE_RESOLUTION, 3] if _T == 1\n'
      '                  else [batch_size, _T, *_model.IMAGE_RESOLUTION, 3])\n'
      '        image_spec = jax.ShapeDtypeStruct(_shape, jnp.float32)\n'
      '        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)')
if "PI05_MEM_FRAMES" in t:
    print("  pi0_config.py: 이미 적용됨")
else:
    assert t.count(A2) == 1, f"pi0_config.py 앵커 {t.count(A2)}개"
    g.write_text(t.replace(A2, N2, 1)); print("  pi0_config.py: 적용")
