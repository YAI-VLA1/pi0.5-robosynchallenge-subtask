# Runpod 이전 견적 — RoboSyn π0.5

작성 2026-10-04 · 측정 환경 VESSL 파드 (RTX 3090 24GB, Ubuntu 20.04)
코드 기준 `YAI-VLA1/pi0.5-robosynchallenge-subtask` 72d7e96 + 재리뷰 수정분

**표기 규칙**
- **[실측]** 이 파드에서 직접 잰 값
- **[추정]** 재보지 않고 계산하거나 외삽한 값. 확정으로 쓰지 말 것
- **[미측정]** 숫자를 낼 근거가 없는 항목

---

## 0. 먼저 — 이전이 "빠르게" 가 아니라 "돌리려면" 필요하다

실제 `train.py` 로 3스텝씩 돌려 본 결과다. **[실측, 전부 3090 24GB]**

| 설정 | 활성화 메모리 | 실패한 할당 | 결과 |
|---|---|---|---|
| batch=1, T=1, CE **on** | 15.90 GiB | 5.58 GiB | **OOM** |
| batch=1, T=1, CE **off** | 15.98 GiB | 5.76 GiB | **OOM** |
| batch=2, T=1, CE **off** | 15.99 GiB | 5.85 GiB | **OOM** |
| batch=1, T=6(MEM), CE on | 16.13 GiB | 5.84 GiB | **OOM** |

**배치·CE·MEM 을 어떻게 바꿔도 15.9~16.1 GiB 로 거의 같다.**
batch 1 → 2 가 더한 것이 **0.01 GiB** 다. 즉 이 구간에서 메모리는
배치가 아니라 **고정 비용**이 지배한다.

### 고정 비용의 정체 **[실측 + 계산]**

`train.py` 는 frozen 파라미터만 bf16 으로 바꾸고 **trainable 은 float32 로 둔다**
(`scripts/train.py` L103–104).

```
frozen   2.94B × 2B(bf16)        5.87 GB
trainable 467M × 4B(fp32)        1.87 GB
optimizer 467M × 2 × 4B          3.74 GB
──────────────────────────────────────
                                11.5 GB
+ 활성화(remat 후)              ~4.5 GB
──────────────────────────────────────
                                ~16 GiB   <- 실측과 일치
```

그리고 실패하는 할당 **5.8 GiB 는 frozen 파라미터 트리(5.87 GB)와 거의 같다.**
학습 스텝 안에서 파라미터 트리 **사본 하나**를 더 잡으려다 터진다 **[추정 — 원인
확정 안 함]**. 버퍼 donation/aliasing 문제일 가능성이 있다.

### 이것이 GPU 선택에 주는 함의

- **최소 VRAM ≈ 16 GiB(상주) + 5.9 GiB(일시) ≈ 22 GiB + 여유** → 24 GB 로는 안 된다
- **배치를 키우는 비용이 이 구간에서 거의 0 이다.** batch 1→2 가 0.01 GiB 였다.
  큰 GPU 에서는 배치를 꽤 올릴 수 있을 가능성이 높다 **[추정 — batch 4 이상은
  이 카드에서 재지 못했다. 외삽하지 말 것]**
- **MEM(T=6) 의 추가 비용도 작다.** T=1 → T=6 이 0.23 GiB(16.13 vs 15.90).
  MEM 때문에 더 큰 GPU 가 필요한 것이 아니다

> **앞서 세웠던 "subtask CE 가 메모리 주범" 가설은 틀렸다.** CE 를 꺼도 같은
> 자리에서 터진다. 원인은 CE 가 아니라 파라미터 dtype 과 파라미터 트리 사본이다.
>
> **이전 문서의 GPU 표(batch 4 에서 10.77 GiB)는 쓰지 말 것.**
> 그 숫자는 `tests/mem_speed.py` 하니스 값이고, 체크포인트를 전부 bf16 으로
> 올리고 optimizer 도 config 것이 아니라 `optax.adamw(1e-5)` 를 썼다.
> 실제 `train.py` 와 **5 GiB 이상 차이**가 난다. 외부 리뷰 R12 지적이 맞았다.

**따라서 24 GB 급에서는 현재 설정으로 학습이 시작되지 않는다.**

---

## 1. `/workspace` 사용량 **[실측]**

전체 **61 GB** (`du` 가 하드링크를 한 번만 세므로 실제 점유량이다).

| 항목 | 크기 | 비고 |
|---|---|---|
| 코드 + venv | **13 GB** | `policy/pi05/.venv`. uv 캐시와 하드링크 공유 |
| 저장소 코드 (venv·데이터 제외) | 1.4 GB | |
| 데이터셋 `cobotmagic_Sim_items_handover` | **8.6 GB** | parquet 1,000 + mp4 3,000 |
| 병합 데이터셋 `_mix` | 14 MB | 데모는 심볼릭 링크, 롤아웃 parquet 98개만 실파일 |
| pi0.5 base 가중치 캐시 `.cache/openpi` | **5.3 GB** | |
| uv 캐시 `.cache/uv` | 24 GB (표시) | **venv 와 하드링크 공유** — 순수 추가분은 약 11 GB |
| 체크포인트 `ckpt_dl` | **18 GB** | 8.9 GB × 2 개 |
| 롤아웃 데이터셋 | 1.0 GB | 실패 100개 (raw + 정렬 복구본) |
| YOLO 라벨 | 30 MB | 1,000 에피소드 |
| HF 캐시 `hf-home` | 510 MB | |
| 로그·중간산출 | ~1 GB | |

### 정리하면 줄어드는 것

| 대상 | 회수량 | 위험 |
|---|---|---|
| `uv cache clean` | **~11 GB** | venv 재생성 시 재다운로드 (약 6분) |
| 체크포인트 1개만 남기기 | **~9 GB** | HF 업로드 확인 후면 안전 |
| `_dl_check`, `fresh_test`, 중간 로그 | ~2 GB | 안전 |
| 롤아웃 raw (정렬 복구본만 보관) | ~0.5 GB | HF 에 raw 가 있으므로 안전 |

**최소 상주 세트 (학습에 반드시 필요한 것) [실측 합산]**

```
코드 + venv      13.0 GB
데이터셋          8.6 GB
pi0.5 base       5.3 GB
───────────────────────
고정             26.9 GB
```

---

## 2. 체크포인트 한 개 **[실측]**

로컬 `ckpt_dl/subtask/checkpoints/81999` = **8.9 GB**

```
params         6.0 GB   모델 가중치
train_state    3.0 GB   optimizer 모멘트 + step   <- 있다
assets          16 KB   norm_stats.json
_CHECKPOINT_METADATA
```

HF 업로드본 (`yai-robosync/pi05-items-handover-subtask-cos82k/checkpoints/81999`) **[실측]**

```
params        6.34 GB  (20 파일)
train_state   3.17 GB  (21 파일)
assets                 norm_stats 포함
합계          9.50 GB
```

**→ `train_state` 까지 올라가 있으므로 HF 에서 받아 full-state resume 이 가능하다.**
`assets` 도 같이 있어 norm_stats 를 따로 챙길 필요가 없다.

### 다만 주의 — `--resume` 과 warm start 는 다르다

완료된 82k 체크포인트를 같은 config 로 `--resume` 하면 `train_state.step` 이
이미 82000 이라 `range(start_step, num_train_steps)` 가 비고 **한 스텝도 안 돈다.**
새 데이터·새 설정으로 더 학습하려면 `mistake/prepare_posttrain_config.py` 가 만드는
**weights-only warm start config** 를 쓴다 (optimizer·step 새로 초기화).
`--resume` 은 **같은 실험의 장애 복구 전용**이다.

---

## 3. 저장 → 업로드 → 삭제 사이클의 최대 디스크 **[실측 + 추정]**

### 코드에서 확인한 동작 **[실측]**

- `checkpoints.py` 의 `max_to_keep=3` → **로컬에 체크포인트 3개가 동시에 남는다**
  (`prepare_repo.py` 가 upstream 의 1 에서 3 으로 올려 놓는다. 업로드가 저장
  간격보다 느릴 때 유실되지 않게 하려던 조치다.)
- `save_interval=1_000` → 1,000 스텝마다 저장
- 업로더는 **학습을 막지 않는다.** `watchdog.sh` 가
  `nohup .venv/bin/python upload_checkpoints.py &` 로 **별도 프로세스**를 띄우고
  60초마다 폴링한다. 학습 루프는 업로드를 기다리지 않는다.
- HF 보관 정책: 5,000(또는 10,000) 배수 + 최근 2개

### 피크 계산

```
보관 중 체크포인트 3개        26.7 GB   [실측 8.9 × 3]
저장 중인 새 체크포인트 1개     8.9 GB   [실측] orbax 가 쓰는 동안
───────────────────────────────────────
체크포인트 피크               35.6 GB
고정 세트                     26.9 GB
───────────────────────────────────────
합계                          62.5 GB
+ 로그·캐시·여유               ~8 GB    [추정]
───────────────────────────────────────
학습 중 실사용 피크           약 70 GB
```

**업로드 실패 시**: 업로더가 멈춰도 `max_to_keep=3` 이 오래된 것을 지우므로
디스크가 무한히 늘지는 않는다. 다만 **지워진 체크포인트는 HF 에도 없으면
복구 불가**다. 업로드 실패가 쌓이면 학습을 멈추고 수동 업로드하는 편이 안전하다.

`max_to_keep=1` 로 줄이면 피크가 **약 44 GB** 로 내려간다. 대신 업로드가
저장 간격 안에 못 끝나면 그 체크포인트를 잃는다.

---

## 4. Network Volume 산정

### 4.1 "기본 설정 그대로" 일 때

| 용량 | 가능 | 제약 |
|---|---|---|
| **50 GB** | ✗ | 고정 26.9 GB 를 빼면 23 GB. 체크포인트 3개(26.7)가 안 들어간다 |
| **80 GB** | △ | `max_to_keep=1` 이면 가능하나 업로드 실패 여유가 없다 |
| **100 GB** | ○ | 기본 `max_to_keep=3` 그대로 피크 ~70 GB |
| **150 GB** | ◎ | 캐시 정리 불필요, 평가 산출물까지 |

### 4.2 줄인 구성 — **50 GB 로 내려간다**

고정 비용을 세 군데서 깎을 수 있다.

**① pi0.5 base 캐시 5.3 GB 를 아예 안 받는다**
기존 81999 에서 warm start 하면 `weight_loader` 가 로컬 체크포인트를 가리킨다.
`gs://openpi-assets/checkpoints/pi05_base` 는 **한 번도 안 쓰인다.**
처음부터 학습할 때만 필요하다.

**② venv 에서 시뮬레이터 의존성이 빠진다 — 약 3.5 GB [추정]**
이 파드의 venv(12 GB)에는 평가용 패키지가 남아 있다:

```
open3d 1.1G · dexsim 693M · vtkmodules 424M · cmeel 380M
warp 356M · pymeshlab 264M · casadi 261M      합계 ~3.5 GB
```

`prepare_repo.py` 가 `pyproject.toml` 에서 `dexsim-engine`/`embodichain`/
`robosynchallenge` 를 빼므로 **새 머신에서 `uv sync` 하면 이것들이 안 깔린다.**
학습 import 체인에 `dexsim` 이 안 뜨는 것도 확인했다 **[실측]**.
→ 학습 전용 venv 는 약 **8.5 GB [추정]**.
(`torch` 는 LeRobot 데이터로더가 쓰므로 남겨야 한다 **[실측]**.)

**③ 설치 후 `uv cache clean` — 11 GB**
venv 가 서고 나면 캐시는 필요 없다. 다시 설치할 때만 재다운로드(약 6분).

**④ seed 체크포인트는 첫 저장 뒤 지운다 — 8.9 GB**
81999 는 가중치를 한 번 읽는 데만 쓴다. 첫 체크포인트가 저장되고 HF 업로드가
확인되면 로컬에서 지워도 된다 (HF 에 원본이 있다).

#### 줄인 구성의 디스크 추이

```
[설치 중 · CPU Pod]
  venv + 코드          10.0 GB   (8.5 + 1.4)
  데이터                8.6 GB
  seed 체크포인트       8.9 GB
  uv 캐시              11.0 GB   <- 설치 끝나면 삭제
  ─────────────────────────────
  설치 피크           ~38.5 GB

[학습 중 · GPU Pod, max_to_keep=1]
  venv + 코드 + 데이터  18.6 GB
  체크포인트 1개         8.9 GB
  저장 중 일시          +8.9 GB
  로그·여유             ~4.0 GB
  ─────────────────────────────
  피크                ~40.4 GB
```

| 용량 | 줄인 구성에서 |
|---|---|
| **50 GB** | ○ **권장 최소.** `max_to_keep=1` · base 캐시 없음 · uv 캐시 정리. 여유 ~10 GB |
| **60 GB** | ◎ **권장.** `max_to_keep=2` 로 업로드 실패 여유까지 (피크 ~49 GB) |
| 80 GB 이상 | 불필요 — 월 $1.4 를 더 내는 값이 없다 |

### 4.3 50 GB 로 가려면 반드시 같이 바꿔야 하는 것

**`max_to_keep=1` 은 그냥 켜면 위험하다.** 업로더는 **원격만** 지우고
(`upload_checkpoints.py` L67–74), 로컬 삭제는 orbax 의 `max_to_keep` 이 한다.
둘 사이에 조율이 없어서, **업로드가 끝나기 전에 orbax 가 그 체크포인트를
지울 수 있다.** 그러면 그 스텝은 영영 사라진다.

upstream 기본값이 1 인데 `prepare_repo.py` 가 3 으로 올려 둔 것도 이 때문이다.

안전하게 가려면 둘 중 하나:

- **(권장) `save_interval` 을 올린다.** 조건은
  `save_interval × step_time  >  업로드 시간 × 2`.
  **빠른 GPU 일수록 이게 빡세진다** — 같은 1,000 스텝이 벽시계로 짧아지는데
  업로드 시간(9.5 GB 전송)은 그대로이기 때문이다.
  H100 에서 step 이 0.5초면 1,000 스텝 = 8분이라 업로드를 못 따라간다.
  **`save_interval=3000~5000` 으로 두는 것이 맞다** (= 체크포인트 개수도 줄어
  HF 저장량도 준다).
- **`max_to_keep=2`** 로 두고 60 GB 를 쓴다. 한 개가 업로드 중이어도 여분이 있다.

**업로드 시간 [부분 실측]**: HF 에서 9.6 GB 다운로드가 깨끗한 회차에 **2분 48초**
(약 57 MB/s) 였다. 끊겨서 재시도한 회차는 13분이었다. 업로드도 같은 자릿수로
보지만 **Runpod 의 네트워크는 재측정이 필요하다 [미측정]**.

### 4.4 더 줄일 수 있는 것 — 코드 변경 필요

**중간 체크포인트는 `params` 만 올린다 (6.34 GB, 33% 감소)**
`train_state`(3.17 GB)는 **크래시 복구용 최신 1개에만** 필요하다. 마일스톤
체크포인트는 평가·warm start 용이라 가중치만 있으면 된다.
업로드 시간도 1/3 줄어 `save_interval` 제약이 완화된다.
→ `upload_checkpoints.py` 수정이 필요하다. **아직 안 바꿨다.**

**Container Disk: 20 GB 권장 [추정]**
코드·venv·데이터·캐시를 전부 `/workspace`(네트워크 볼륨)에 두므로 컨테이너
디스크에는 OS + CUDA 런타임 이미지만 남는다. 기본 10 GB 는 JAX/CUDA 이미지에
빠듯할 수 있어 20 GB 를 둔다. (Pod 종료 시 사라진다.)

> **주의: 네트워크 볼륨에서의 I/O.** 체크포인트 8.9 GB 쓰기가 1,000 스텝마다
> 일어난다. 네트워크 볼륨이 로컬 NVMe 보다 느리면 저장이 학습을 늦출 수 있다.
> **[미측정]** — 첫 Pod 에서 `save_interval` 한 번의 실제 소요를 재 볼 것.
> 느리면 `save_interval` 을 2,000~5,000 으로 올린다.

---

## 5. H100 SXM 80GB vs H200 141GB

### 확정할 수 있는 것

| | 값 | 근거 |
|---|---|---|
| 측정 GPU | RTX 3090 24GB | **[실측]** |
| 현재 설정에서 3090 | batch=1 에서도 **OOM** | **[실측]** |
| 활성화 메모리 (batch/CE/MEM 무관) | 15.9~16.1 GiB | **[실측]** 4조합 |
| batch 1 → 2 증가분 | **0.01 GiB** | **[실측]** |
| T=1 → T=6(MEM) 증가분 | **0.23 GiB** | **[실측]** |
| 일시 할당 (파라미터 트리 사본) | 5.6~5.9 GiB | **[실측]** |
| frozen bf16 + trainable fp32 + optimizer | 11.5 GB | **[추정]** 계산값, 위 실측과 정합 |
| **현재 설정 최소 VRAM** | **약 24~28 GB 이상** | **[추정]** 16.1 + 5.9 + 여유 |
| **권장 VRAM** | **40 GB 이상** | **[추정]** 배치를 올리려면 |

### 확정할 수 **없는** 것 — 측정 안 됨

- **H100/H200 의 step time** — 두 GPU 다 돌려 본 적이 없다. 3090 대비 배수를
  추정해 비용을 내면 틀린 숫자가 된다.
- **H100/H200 에서의 최대 batch** — batch 1→2 가 0.01 GiB 였다는 것만 안다.
  batch 4 이상은 이 카드에서 재지 못했다. "남는 VRAM ÷ 0.01 GiB" 같은 외삽은
  하지 말 것 — 활성화는 어느 지점부터 선형으로 늘어난다.
- **MEM T=6 의 전체 비용** — 메모리는 +0.23 GiB 로 작지만(실측), **step time**
  증가는 재지 못했다 (T=1 기준이 OOM 이라 비교 불가).

### mem_speed 와 실제 학습의 차이 **[실측으로 확인됨]**

| | `tests/mem_speed.py` | 실제 `train.py` |
|---|---|---|
| 파라미터 dtype | 전부 bf16 복원 | 모델 초기화 dtype 에 맞춰 병합, frozen 만 bf16 |
| optimizer | `optax.adamw(1e-5)` | config 의 optimizer (clipping, b2 등) |
| CE 경로 | 없음 | `PI05_SUBTASK_W>0` 이면 포함 |
| batch 4, T=1 활성화 | 10.77 GiB | batch 2 에서 이미 16.24 GiB |

**→ `mem_speed.py` 숫자로 production batch 상한을 정하지 말 것.**

---

## 6. 비용 — 구조만, 금액은 조건부

### Runpod 공시 요금 (2026-10 기준, 검색으로 확인)

| 항목 | Community | Secure |
|---|---|---|
| H100 SXM 80GB | $2.69/hr | $3.49/hr |
| H200 SXM 141GB | $3.59/hr | $4.59/hr |
| Network Storage (표준, 1TB 미만) | $0.07/GB/월 | |
| Container Disk | $0.10/GB/월 | |

### 저장소 비용 **[확정 계산 가능]**

| 볼륨 | 월 요금 | 7일 |
|---|---|---|
| **50 GB** | **$3.50** | **$0.82** |
| **60 GB** | **$4.20** | **$0.98** |
| 80 GB | $5.60 | $1.31 |
| 100 GB | $7.00 | $1.63 |
| 150 GB | $10.50 | $2.45 |

Container Disk 20 GB = $2.00/월 (Pod 가 떠 있는 동안).

**저장소는 전체 비용에서 미미하다.** 100 GB 와 50 GB 의 7일 차이는 **$0.81**,
H100 **18분** 요금이다.

그래서 "줄일 수 있으니 줄인다" 보다 **"어느 쪽이 체크포인트를 잃을 위험이
낮은가"** 로 정하는 편이 맞다. 60 GB 면 `max_to_keep=2` 로 업로드 실패 여유가
생기고, 100 GB 대비 7일 **$0.65** 차이다.

### GPU 비용 **[계산 불가 — throughput 미측정]**

82k 스텝이 몇 시간인지 모른다. 3090 의 1.44초/스텝은 벤치마크 하니스 값이고
실제 `train.py` 값이 아니다(그리고 그 설정은 3090 에서 돌지도 않는다).

**숫자를 내려면 H100 에서 1시간 측정이 선행되어야 한다.** 그 1시간이
Community H100 기준 **$2.69** 다.

### 체크포인트 저장·업로드 시간 **[부분 실측]**

- 업로드 9.5 GB. 이전 파드에서 HF 다운로드가 끊김 재시도 포함 약 13분 걸렸다 **[실측]**.
  업로드도 비슷한 규모로 본다 **[추정]**.
- **업로드는 학습을 막지 않는다** (별도 프로세스) — GPU 요금에 더해지지 않는다.
- 저장 자체(8.9 GB 쓰기)는 학습을 멈춘다. 네트워크 볼륨 속도에 달려 있다 **[미측정]**.

---

## 추천

### A. 비용 최소

```
Network Volume    60 GB        $4.20/월  ($0.98 / 7일)
Container Disk    20 GB        $2.00/월
준비              CPU Pod      데이터 + venv + 81999 (GPU 요금 0)
                               base 캐시 받지 않음, 설치 후 uv cache clean
설정              max_to_keep=2, save_interval=3000
측정              H100 Community 1시간   $2.69
본학습            H100 Community         측정 후 결정
```

- **50 GB 도 된다** (`max_to_keep=1`). 7일에 $0.16 더 싸다. 그 $0.16 때문에
  업로드가 밀렸을 때 체크포인트를 잃을 위험을 지는 것은 권하지 않는다.
- **100 GB 는 불필요하다.** base 캐시(5.3) + uv 캐시(11) + 시뮬레이터
  패키지(3.5) + 체크포인트 2개(17.8) = 37.6 GB 가 줄일 수 있는 몫이다.

### B. 학습 시간 우선

```
Network Volume   100 GB        $7.00/월   (max_to_keep=3 그대로, 설정 안 건드림)
Container Disk    20 GB
GPU              H100 80GB     $2.69/hr (Community)
```

- 다만 **H200 을 고를 근거가 약해졌다.** 고정 비용이 ~22 GiB 로 확인됐으니
  H100 80GB 에도 **58 GB 가 남는다.** 배치 증가 비용이 이 구간에서 거의 0 인
  것까지 보면, 메모리가 병목일 가능성은 낮다.
- **H200 이 H100 보다 몇 배 빠른지는 측정값이 없다.** 시간당 33% 비싼데
  33% 이상 빠르다는 근거가 지금은 없다. **H100 으로 시작하는 쪽을 권한다.**

### 두 구성 모두에 공통 — 먼저 할 것

1. **CPU Pod 에서 `setup_train.sh` 완주.** fresh worktree 로 검증해 뒀다
   (`SETUP.md` §2). 데이터 8.6 GB + venv + base 5.3 GB 를 여기서 받는다.
2. **GPU Pod 1시간으로 측정만.** 실제 `train.py` 로
   `batch 1/2/4/8 × (CE on/off) × (T=1/6)` 의 peak VRAM 과 step time 을 잰다.
   `smoke_matrix.sh` 가 그 형태로 되어 있다.
3. **그 숫자로 본학습 GPU·시간·비용을 정한다.**

**2번을 건너뛰고 장시간 예약하지 말 것.** 지금 가진 것으로는 H100 의
batch 상한도, 82k 소요 시간도 모른다.

---

## 아직 조사 중

- **파라미터 트리 사본 5.8 GiB 를 없앨 수 있는가.** 버퍼 donation 이 안 걸린
  것이라면 고칠 여지가 있고, 그러면 24 GB 급에서도 돌 수 있다. 다만 어차피
  Runpod 으로 가면 **속도** 때문에라도 큰 카드가 낫다.
- 네트워크 볼륨에서의 체크포인트 저장 소요 시간
- 저장 → HF 업로드 → 로컬 삭제 한 사이클의 실제 벽시계 시간
- **H100 에서의 batch 상한과 step time** — 이게 없으면 본학습 시간·비용을
  확정할 수 없다. 1시간 측정으로 끝난다.

## 출처

- [Runpod GPU Cloud Pricing](https://www.runpod.io/pricing)
- [NVIDIA H100 Price (October 2026) — Runpod](https://www.runpod.io/articles/guides/nvidia-h100)
- [H100 vs H200 — Runpod](https://www.runpod.io/articles/comparison/nvidia-h200-vs-h100-choosing-the-right-gpu-for-massive-llm-inference)
- [RunPod pricing in 2026: GPU Pods, Serverless, and storage](https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs)
