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
