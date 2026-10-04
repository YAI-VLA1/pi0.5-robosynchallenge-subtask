import pathlib, sys
# 경로 계약: 인자 1개 = <repo>/policy/pi05 (PI05 루트). 생략하면 VESSL 기본값.
# 예전에는 이 값을 상수로 박아 둬서 다른 머신에서 apply_all.sh 가
# 엉뚱한 checkout 을 고치거나 FileNotFoundError 를 냈다. (리뷰 R7)
_PI05 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                     else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
p = _PI05 / "src/openpi/models/model.py"
s = p.read_text()
A = """    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,"""
N = """    # ── MEM: 히스토리 드롭아웃 (논문 0.3) ──────────────────────────────
    # 학습 때 히스토리 전체를 통째로 떨궈 모델이 과거에 과의존하지 않게 한다.
    # 샘플마다 결정하고 **카메라 전체에 같은 결정**을 적용한다 (한 카메라만
    # 과거가 있으면 시간 정렬이 어긋난다). 현재 프레임은 항상 남긴다.
    out_frame_valid = observation.frame_valid
    if train and out_frame_valid is not None:
        _p = float(os.environ.get("PI05_MEM_HIST_DROPOUT", "0.3"))
        if _p > 0:
            _rng, rng = jax.random.split(rng)
            _any = next(iter(out_frame_valid.values()))
            _b, _t = _any.shape[0], _any.shape[-1]
            _drop = jax.random.bernoulli(_rng, _p, (_b,))          # 샘플별, 카메라 공통
            _only_cur = jnp.zeros((_t,), dtype=bool).at[-1].set(True)
            out_frame_valid = {
                k: jnp.where(_drop[:, None], _only_cur[None, :], v)
                for k, v in out_frame_valid.items()
            }

    return Observation(
        images=out_images,
        image_masks=out_masks,
        frame_valid=out_frame_valid,
        state=observation.state,"""
if "PI05_MEM_HIST_DROPOUT" in s:
    print("이미 적용됨")
else:
    assert s.count(A) == 1, f"앵커 {s.count(A)}개"
    s = s.replace(A, N, 1)
    if "\nimport os\n" not in s:
        s = s.replace("import dataclasses\n", "import dataclasses\nimport os\n", 1)
    p.write_text(s); print("OK — 히스토리 드롭아웃 0.3 (카메라 공통, 현재는 항상 유지)")
