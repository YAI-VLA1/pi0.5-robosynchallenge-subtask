# 백업 상태 — 2026-10-05 파드 중단 전

## HuggingFace (영속)

`yai-robosync/handover-rollouts`

| 경로 | 내용 |
|---|---|
| `mem_ep1_10661/` | **최종 체크포인트 롤아웃 100개** (959 MB). state/qvel/qf/action, pen_pose/holder_pose, 3뷰 영상, LeRobot v2.1. `stage_metrics.json`·`mistake_labels.json`·`rollout_meta.json` 포함. **정렬 맞음**(RSC_ALIGN 이후 수집) |
| `subtask/` | 기존 subtask 체크포인트 실패 롤아웃 100개. **정렬 안 맞음** — `rollout/fix_rollout_alignment.py` 필요 |
| `videos/mem_ep1_7000/` | step 7000 롤아웃 30개 + subtask 오버레이 |
| `videos/mistake_windows/` | mistake 구간 표시 영상 9개 |
| `videos/obb_check/` | bbox 투영 GT 검증 그림 |

`yai-robosync/pi05-items-handover-subtask-mistake-mem-ep1` — 체크포인트 1000~10661

## GitHub (영속)

`YAI-VLA1/pi0.5-robosynchallenge-subtask` — `017fbe5`
코드·패치·테스트·문서 전부. `SETUP.md` 로 새 머신 재현 가능.

## /root/rsc_recover (파드 재시작 생존)

```
labels/bbox_tokens.tgz     bbox 4토큰 라벨 1,000 에피소드 (614 KB)
labels/obb_labels.tgz      회전 박스 라벨 1,000 에피소드 (3.6 MB)
snapshots/config_with_bbox.py          bbox 배선된 config.py
snapshots/stage_metrics.json           10661 단계별 지표
snapshots/mistake_labels.json          10661 mistake 라벨
snapshots/rollout_meta.json            seed/success
snapshots/subtask_gen_..._n100.jsonl   생성된 subtask 토큰 덤프
snapshots/probe*.log                   학습 설정 역추적 결과
snapshots/smoke_matrix.log             실제 train.py 메모리 측정
RUNPOD_PLAN.md · BBOX_PLAN.md          견적·설계 문서
gh_subtask/                            저장소 사본
```

## 복구 순서

```bash
bash /root/rsc_recover/gh_subtask/setup/setup_train.sh --all   # 환경
tar xzf /root/rsc_recover/labels/bbox_tokens.tgz -C /workspace # bbox 라벨
tar xzf /root/rsc_recover/labels/obb_labels.tgz -C /workspace
python3 /root/rsc_recover/gh_subtask/patches/verify_patches.py <repo> --all
```

롤아웃 데이터는 HF 에서 받는다. 체크포인트도 HF 에 있다.

## 안 올린 것

- `/workspace/mem_ep1_vid_10661/` — 10661 롤아웃 영상 100개 (생성 중이었다).
  재생성 가능: `subtask_video.py <롤아웃> <덤프> <출력>`. 덤프는 snapshots 에 있다.
- venv·데이터셋 8.6 GB — `setup_train.sh` 가 다시 받는다.
