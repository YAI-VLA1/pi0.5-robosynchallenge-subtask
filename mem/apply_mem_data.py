#!/usr/bin/env python3
"""학습 데이터 로더가 과거 프레임을 함께 뽑게 한다 (MEM 입력).

LeRobot 의 delta_timestamps 를 쓴다 — 이미 action chunk 에 쓰고 있는 기능이다.
이미지 키에 {-(T-1)*stride, ..., -stride, 0} 초를 요청하면
  · 프레임이 (T,H,W,C) 로 쌓여 나오고
  · `{key}_is_pad` 불리언이 같이 나온다 → 그게 곧 frame_valid 다
에피소드 시작부의 부족한 과거는 LeRobot 이 패딩하고 is_pad 로 표시한다.
다른 에피소드 프레임을 끌어오지 않는다.

PI05_MEM_FRAMES (기본 1 = 끔) · PI05_MEM_STRIDE_S (기본 1.0)
"""
import pathlib, sys

F = "src/openpi/training/data_loader.py"

A = '''    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )'''
N = '''    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    delta = {
        key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }
    # ── MEM: 과거 프레임을 같이 뽑는다 ──────────────────────────────────
    # PI05_MEM_FRAMES=6, PI05_MEM_STRIDE_S=1.0 이면 -5,-4,-3,-2,-1,0 초.
    # LeRobot 이 {key}_is_pad 를 같이 주므로 frame_valid 를 따로 만들 필요가 없다.
    import os as _os
    _T = int(_os.environ.get("PI05_MEM_FRAMES", "1"))
    if _T > 1:
        _stride = float(_os.environ.get("PI05_MEM_STRIDE_S", "1.0"))
        _offsets = [-(_T - 1 - i) * _stride for i in range(_T)]   # oldest -> current(0.0)
        _img_keys = [k for k in dataset_meta.features
                     if k.startswith("observation.images.")]
        for k in _img_keys:
            delta[k] = _offsets
        print(f"[MEM] 히스토리 {_T} 프레임 · 간격 {_stride}s · offsets {_offsets}", flush=True)
        print(f"[MEM] 대상 이미지 키 {_img_keys}", flush=True)
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, delta_timestamps=delta)'''


def main():
    p = pathlib.Path(sys.argv[1]) / F
    s = p.read_text()
    if "PI05_MEM_FRAMES" in s:
        print("이미 적용됨"); return
    if s.count(A) != 1:
        sys.exit(f"앵커가 {s.count(A)}개 — 중단")
    p.write_text(s.replace(A, N, 1))
    print(f"패치 완료 {p}")


if __name__ == "__main__":
    main()
