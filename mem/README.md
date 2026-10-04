# MEM short-term video memory — JAX/Flax 이식

pi0.5 의 SigLIP 비전 타워에 MEM(Multi-Scale Embodied Memory)의 단기 영상 메모리를
얹는다. PI 공식 구현·가중치는 공개돼 있지 않다 — **이식이지 재현 주장이 아니다**.

근거: MEM 논문 arXiv 2603.03596 §III-C / Appendix C, π₀.₇ arXiv 2604.15483,
Knowledge Insulation arXiv 2505.23705.

## 핵심 성질

새 학습 파라미터가 **0개**다. 공간 어텐션의 Q/K/V/W_O 를 시간축에 재사용하고,
시간 임베딩은 고정 sinusoidal 이다. 따라서 **기존 pi0.5 체크포인트가 그대로 로드된다**.

## 합성 (§5 — 이걸 틀리면 MEM 이 아니다)

공간 블록과 시간 블록을 따로 쌓지 **않는다**. Q/K/V 를 한 세트만 뽑고,
시간 어텐션으로 V 를 섞은 뒤 그 V 로 공간 어텐션을 하고, W_O 를 한 번만 건다.

```
U       = LN(Z) + e(t)              # e 는 temporal 층의 QKV 입력에만
Q,K,V   = W_Q(U), W_K(U), W_V(U)
V_time  = causal_softmax_t(Q·Kᵀ) V  # 같은 패치의 과거만
Y       = softmax_p(Q·Kᵀ) V_time    # 프레임 내부
Z       = Z + W_O(Y) ; Z = Z + MLP(LN2(Z))
```

공간→시간 두 블록으로 만들면 W_O 와 residual 이 두 번 걸려 **T=1 에서도 값이 달라진다**.
올바른 합성이면 T=1 에서 시간 softmax 가 identity 라 특수 처리 없이 환원된다.

## 설정

| | 값 | 근거 |
|---|---|---|
| temporal 층 | `(i+1) % 4 == 0` → depth 27 에서 0-based 3/7/11/15/19/23 | 논문 |
| 시간 임베딩 | 상대 인덱스 `[-(T-1)..0]`, sin + (cos−1), base 10000, **e(0)=0** | 논문 + LeRobot |
| 과거 토큰 | 인코더 출력에서 폐기 → 토큰 수가 T 와 무관 | 논문 |
| 사전학습 설정 | T=6 (과거 5 + 현재), 1초 간격 | 논문 |

`scan=True` 체크포인트라 층 인덱스를 알 수 없어, 27칸 불리언을 scanned input 으로
넘기고 `lax.cond` 로 분기한다 (`jnp.where` 는 양쪽을 다 계산해 4.5배 낭비).

## 적용

```bash
python3 mem/apply_mem.py <repo>/policy/pi05
```
앵커 9개 전부 유일성 검사를 거친다. 업스트림이 바뀌면 조용히 틀리는 대신 멈춘다.

## 검증 (tests/)

| 스크립트 | 확인 |
|---|---|
| `mem_equiv.py` | T=1 에서 패치 전후 비트 동치 |
| `mem_t_tests.py` | causal · padding 무영향 · 과거→현재 전달 · 토큰 수 · 파라미터 · NaN |
| `mem_t_f64.py` | float32 누적오차인지 논리 버그인지 판별 |
| `mem_image_path.py` | 전체 이미지 경로 ((B,T,H,W,3) 입력, 과거 토큰 폐기) |
| `mem_cost.py` | T=1 vs T=6 추론 비용 |

### 검증 결과 (CPU, float32, subtask 체크포인트 81999)

| 항목 | 값 |
|---|---|
| T=1 비트 동치 (실제 모델 경로) | **0.000e+00** |
| 이미지 경로: 현재만 valid == T=1 | **0.000e+00** |
| causal — 미래→과거 누수 | **0.000e+00** |
| padding 프레임 무영향 | **0.000e+00** |
| 과거→현재 전달 (토큰 / 이미지) | 2.288 / 1.000 |
| 현재만 valid == T=1 (토큰 수준) | 1.5e-5 (f32) → 2.3e-14 (f64) = 누적오차 |
| 토큰 수 · 파라미터 키 · NaN | 통과 |
| 비용 배율 T=6/T=1 | 3.09x (CPU; GPU 절대값은 미측정) |

## 아직 안 된 것

- 학습 데이터 로더에서 과거 프레임 샘플링 (LeRobot `delta_timestamps` + `_is_pad`)
- 추론 프레임 버퍼 (에피소드 경계 초기화, 과거 부족 시 invalid 표시)
- 히스토리 드롭아웃 (논문 0.3)

## 학습 경로 배선 (데이터 -> 모델)

| 파일 | 패치 | 내용 |
|---|---|---|
| `data_loader.py` | `apply_mem_data.py` | `delta_timestamps` 로 과거 프레임 + `_is_pad` |
| `config.py` / `libero_policy.py` | `apply_mem_wire2.py` | repack 이 `_is_pad` 통과, `frame_valid` 생성 |
| `model.py` | `apply_mem_obs.py` | `Observation.frame_valid`, `from_dict` 전달, 타입 주석 분리 |
| `pi0.py` / `pi0_config.py` | `apply_mem_numframes.py` | 비전 모듈에 `num_frames`, fake_obs 모양 |
| `image_tools.py` | `apply_mem_resize.py` | `resize_with_pad` 를 임의 선행 차원으로 |
| `model.py` | `apply_mem_aug.py` | 증강을 프레임 공통으로 (§9) |
| `model.py` | `apply_mem_crop.py` | `RandomCrop` 치수를 뒤에서 세도록 |

전부 `PI05_MEM_FRAMES=1`(기본)에서 기존과 동일하게 동작한다.

### 1스텝 실행에서 잡은 버그 6개

shape 출력만 봐서는 안 드러나고, 실제로 forward/backward 를 돌려야 나온 것들이다.

| 버그 | 증상 |
|---|---|
| `x_2d` reshape 가 과거 토큰 폐기 전 배치 사용 | 채널 1152 -> 384 |
| `from_dict` 가 `frame_valid` 누락 | 패딩을 진짜 과거로 착각 |
| `Observation` 타입 주석 충돌 | `*b` 가 `(4,6)` vs `(4,)` |
| `resize_with_pad` 가 5차원 미지원 | 차원을 하나 더 붙임 |
| `augmax` 증강이 프레임마다 독립 | 같은 patch 의 시간 대응 훼손 |
| **`RandomCrop` 이 `(T,H)` 를 `(H,W)` 로** | **cam_high 만 5px 로 뭉개짐, 예외 없음** |

마지막 둘은 예외가 안 나므로 학습을 한참 돌린 뒤에야 드러났을 것이다.

### 데이터 로더 검증

```
이미지            (6, 3, 480, 640) -> 변환 후 (6, 224, 224, 3)
에피소드 시작      is_pad [1,1,1,1,1,0]
에피소드 중간      is_pad [0,0,0,0,0,0]
현재와의 평균차    0.29 0.27 0.25 0.31 0.26 0.00   (같은 프레임 복제가 아니다)
비전 파라미터      414,803,696  (MEM 적용 전과 동일)
```

### 학습 1스텝 (CPU, T=2, batch 1)

```
입력          image (1,2,224,224,3) x 3대 · frame_valid (1,2) [0,1]
비전 파라미터  414,803,696        (MEM 적용 전과 동일 = 새 파라미터 0)
loss          0.051901  유한
grad norm     1.046875  NaN/Inf 없음
```

frame_valid 가 `[0,1]` 이라 과거가 전부 막힌 상태인데도 loss 가 유한하다 —
전부 막힌 softmax 에서 자기 자신을 열어 두는 안전장치가 작동한다.
그래디언트가 흐르므로 시간 어텐션 경로가 역전파에서 끊기지 않았다.

### 아직 안 된 것

- 추론 프레임 버퍼 (에피소드 경계 초기화, 과거 부족 시 invalid)
- 히스토리 드롭아웃 (논문 0.3)
- GPU 에서 T=6 학습 1스텝 / 비용 실측 — 롤아웃 수집이 GPU 를 쓰는 중이라 대기
