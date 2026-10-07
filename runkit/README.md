# runkit — 끊겨도 스스로 돌아오는 학습

장시간 학습을 사람이 안 보는 동안 돌리기 위한 것이다. 세 가지를 한다.

1. **파드가 죽어도 복구된다** — `/workspace` 가 통째로 비워지는 환경을 전제한다.
   `bootstrap_bbox.sh` 한 줄로 venv·데이터·라벨·병합 데이터셋·체크포인트를
   전부 다시 세우고 HuggingFace 의 최신 체크포인트에서 이어 간다. 멱등이다.
2. **학습이 죽어도 되살아난다** — `run_training.sh` 가 최대 20번 `--resume`
   으로 재시작한다. 단 2분 안에 죽는 게 3번 연속이면 환경 문제로 보고 멈춘다
   (설정이 깨진 채로 20번을 태워봐야 의미가 없다).
3. **백업이 끊기면 알아챈다** — `watchdog.sh` 가 60초마다 업로더 생존을 본다.
   생존 확인은 `pgrep -f` 가 아니라 **PID 파일**로 한다 — 그 문자열을 담은
   다른 명령줄(진단 명령, 워치독 자신)까지 매칭해 죽은 업로더를 살아 있다고
   오판한 적이 있다.

영속 저장소는 HuggingFace 하나뿐이다. 로컬 디스크는 언제든 사라진다고 본다.

## 파일

| | |
|---|---|
| `bootstrap_bbox.sh` | 진입점. 빈 머신 -> 학습 재개까지. 멱등 |
| `run_training.sh` | 학습 감독 (재시작·디스크 선검사·`BATCH` 덮어쓰기) |
| `watchdog.sh` | 업로더 감시 |
| `upload_checkpoints.py` | 체크포인트 -> HF, 오래된 것은 원격에서 삭제 |
| `fetch_seed.py` | warm-start 체크포인트 + obb 라벨 |
| `fetch_run_ckpt.py` | 이 런의 최신 체크포인트 (없으면 0부터) |
| `fetch_rollouts.py` | 실패 롤아웃 98개 |
| `fetch_urdf.sh` | 손목 FK 용 로봇 URDF |

## 쓰는 법

```bash
bash runkit/bootstrap_bbox.sh                 # 그냥 돈다
BATCH=4 POST_STEPS=20000 bash runkit/bootstrap_bbox.sh
SKIP_TRAIN=1 bash runkit/bootstrap_bbox.sh    # 환경만 세우고 안 띄운다
```

경로는 스크립트 머리에 모여 있다. 다른 머신(Runpod 등)에 올릴 때는 `REC`·`WS`·
`CKPT_BASE` 와 HF 저장소 이름만 보면 된다.

## 주의

- **파드 init script 에 걸어야 진짜 자동이다.** 이 컨테이너에는 cron 이 없다.
  걸 수 없으면 복귀 후 손으로 한 번 돌린다.
- `PI05_*` 환경변수는 **학습과 추론에 같은 값**을 줘야 한다. 프롬프트 형식이
  달라지면 조용히 안 죽고 loss 만 튄다.
- 체크포인트는 `/workspace` 에 쓴다. `/root` 는 용량이 남아도 inode 할당이
  막혀 orbax 가 ENOSPC 로 죽은 적이 있다.
