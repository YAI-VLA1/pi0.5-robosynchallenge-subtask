# 2026-09-25 step 2 — pi0.5 에 subtask 문장 생성 붙이기

step 1 (관측에서 phase 가 읽히는가) 은 통과했다 → `findings.md`.
여기는 구현 기록이다.

## openpi 의 pi0.5 는 논문의 pi0.5 가 아니다

`models/pi0.py` 의 `compute_loss` 는 flow matching MSE **하나뿐**이다.
`prefix_out` 을 계산해놓고 버린다. logits head 도, CE 도, 텍스트 출력 경로도 없다.
즉 **논문의 subtask 예측 부분이 미세조정 코드에 구현돼 있지 않다.**

`pi0_fast` 에는 있지만 **`gemma_fast.py` 라는 별도 모듈**을 쓴다. pi0.5 가 쓰는
`gemma.py` 에는 `return_prelogits` 도 `decode` 도 없고 호출 시그니처도 다르다.
"복사해 붙이면 된다"고 처음에 말한 것은 틀렸다.

쓸 수 있는 것은 `gemma.py:153` 의 `Embedder.decode` 뿐이다 — 입력 임베딩 테이블을
전치해 쓰므로 **logit head 에 새 파라미터가 없다**. `Module` 에 3줄로 노출하면 된다.

## 설계

```
State: {이산화 state}; Subtask: {슬롯 14토큰};\nAction:
                                ^^^^^^^^^^^^ 여기에만 CE
```

* 슬롯은 **jit 용 고정 버퍼**다. 클래스 인덱스가 아니라 매 자리에서 vocab 257,152개
  중 하나를 뽑는 진짜 자연어 생성이다. 길이는 EOS 가 정한다.
* **State 가 Subtask 앞에 와야 한다.** 슬롯이 causal 이라 생성 시점에 state 를 봐야 한다.
* **원본 Task 지시문은 뺐다.** 에피소드 내내 고정이라 정보가 0 인데 22토큰을 먹는다.
  → 다태스크로 갈 때는 되살려야 한다. subtask 어휘가 태스크 간에 겹친다.
* GT 문장 15개는 손으로 썼다 (`sentences.py`). 구간 이름(`left_close0`)을 그대로 쓰면
  PaliGemma 의 언어 prior 가 안 붙는다.

## ★ 함정 1 — prefix 가 완전 양방향이다

`embed_prefix` 의 `ar_mask += [False] * tokenized_inputs.shape[1]`.
그대로 두면 위치 p 의 hidden state 가 **이미 토큰 p+1 을 본다**. 다음 토큰 예측 CE 가
'복사'로 풀리는 자명한 과제가 된다.

실제로 그렇게 됐다: **step 0 의 7.29 가 800스텝 만에 0.046 으로 무너졌다.**

고침: `make_attn_mask` 가 샘플별 마스크(`bool[?B, N]`)를 받고 `cumsum(ar_mask)` 로
블록을 나눈다. **슬롯 자리만 True** 로 두면
* 이미지/State (cumsum 0) 는 슬롯을 못 본다 — 맞다, 생성 대상이다
* 슬롯 k 번째는 0..k 만 본다 — causal
* 뒤의 `;\nAction:` 은 슬롯 전체를 본다 — 조건화에 필요하다

검증 (`/workspace/causal_test.py` 패턴): 슬롯 마지막 토큰을 바꿨을 때
앞쪽 Δhidden = **0.000000**, 직후 tail Δ = 0.055. 의도한 구조다.

부작용: `prefix_ar_mask` 가 1-D → (b, s) 가 되어 `compute_loss` 의
`jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)` 이 깨진다. 같이 고쳐야 한다.
(`sample_actions` 쪽 두 곳은 `make_attn_mask` 가 브로드캐스트해서 무관하다.)

## ★ 함정 2 — 패딩까지 CE 를 걸면 40% 가 낭비다

문장+EOS 가 5~13토큰, 슬롯 14 → **패딩 39.5%**. 손실의 40% 가 "0 을 맞혀라"였다.
EOS(id 1)로 끝내고 `cumsum(eos) - eos == 0` 인 자리만 학습한다.

`token_loss_mask` 는 **슬롯 전체를 덮은 채로 둔다** — attention 의 causal 블록
경계로도 쓰이므로 여기서 줄이면 구조가 깨진다. 손실 쪽에서만 따로 자른다.

## ★ 함정 3 — 전체 정확도는 부풀려진 숫자다

step 1000 (수정 전 버전) 측정: CE 2.36 인데 토큰 정확도 **98.66%**. 모순이 아니다.
teacher forcing 이라 `"close the left gripper on the ___"` 에서 `pen` 은 쉽다.
판단은 **첫 토큰**에 몰려 있다.

→ `subtask_acc_first` 를 따로 찍는다. 이게 "장면을 보고 phase 를 아는가" 지표다.
   step 0 에서 acc 0.32 / acc_first 0.00 으로 갈린다.

## 손실 분리 로깅

`compute_loss(..., return_parts=True)` → `(total, {flow, subtask_ce, subtask_acc,
subtask_acc_first})`, `train.py` 는 `nnx.value_and_grad(..., has_aux=True)`.
`compute_loss` 에 런타임 typecheck 가 없어(`@override` 만) 반환형을 바꿔도 안전하다.
`sow` 는 nnx transform 경계에서 조용히 실패할 여지가 있어 쓰지 않았다.

## 가중치 균형

step 0 에서 CE 6.34 vs flow 0.296 (21:1). step 1000 측정(수정 전)에서 flow 0.047 로
baseline 궤적과 비슷했다 — CE 가 action 학습을 망치지는 않는다.
그래도 `loss/flow` 를 baseline 과 같은 step 에서 비교할 것. 나빠지면 `PI05_SUBTASK_W` 를
0.02 쯤으로 낮춘다 (기본 1.0, 0 이면 완전히 꺼진다).

## 학습되는 파라미터 (측정)

| | 파라미터 |
|---|---|
| 동결 LLM base (PaliGemma 2B) | 2,508,531,712 |
| 동결 LLM base (action expert 300M) | 427,932,672 |
| **학습** SigLIP 비전 타워 | **414,803,696** |
| **학습** LoRA (PaliGemma 2B) | 27,869,184 |
| **학습** LoRA (action expert 300M) | 22,118,400 |
| **학습** proj / MLP 등 | 2,165,792 |

전체 34.0억 중 4.67억(13.7%)이 학습 대상.
**비전 타워가 학습 예산의 89%** 다 — `img` 경로에 `llm` 이 없어 freeze filter 에 안 걸린다.
2B 전체 해제는 이 GPU(3090 24GB)에서 불가능하다(AdamW 상태만 23GB).
언어 용량이 병목이면 LoRA rank 를 16 → 32/64 로 올리는 것이 현실적인 노브다.

## 도구

```
subtask/sentences.py              GT 문장 15개
subtask/apply_subtask_patch.py    tokenizer/transforms/policy/pi0/gemma 패치 (멱등, --check)
subtask/apply_decode_patch.py     추론 디코드 (작성 완료, 미적용 — 평가 전에 적용)
subtask/prepare_subtask_config.py 데이터 배선 + TrainConfig
subtask/eval_loss_parts.py        체크포인트에서 flow/CE/정확도 분리 측정
```

## 운영 메모

* `replace(..., 1)` 이 엉뚱한 클래스를 잡는다 — `config.py` 는 `LeRobotLiberoDataConfig`,
  `libero_policy.py` 는 `LiberoInputs` 에 들어갔다. **앵커 유일성 검사를 넣었다.**
* `Group.push` 는 뒤에 붙는다. `InjectSubtask` 를 data_transforms 에 넣으면
  `EmbodiChainInputs` 뒤가 되고 거기서 `frame_index` 가 이미 사라진 뒤다. → repack 그룹.
* `@dataclasses.dataclass(frozen=True)` 가 이중으로 붙으면
  `Cannot overwrite attribute __setattr__`. 클래스를 끼워 넣을 때 원본 데코레이터 위치를 볼 것.
* **`pkill -f "[w]atchdog.sh"` 가 같은 명령줄에서 watchdog 을 띄우면 자기 셸을 죽인다.**
  브래킷 트릭은 패턴이 그 줄의 유일한 출현일 때만 통한다. 죽이기와 띄우기를 분리할 것.
* wandb 팀 `YAI-VLA` 는 멤버인데도 API 키로 쓸 수 없다(프로젝트명 무관). 권한 문제.
  개인 엔티티 `robosyn333-yai` 로 돌린다 — baseline 도 거기 있어 비교에는 낫다.

## flow 비교 기준선 (2026-09-25)

`eval_loss_parts.py`, CPU, 배치 6개, 각자 학습된 프롬프트 형식으로 측정.

| 런 | step | flow |
|---|---|---|
| baseline (`pi05-items-handover-cos82k`) | 10000 | **0.00421** |
| subtask | 10000 | (측정 예정) |

**wandb 의 `loss/flow` 와 직접 비교하면 안 된다** — 그쪽은 학습 중 GPU·bf16·랜덤 time
추출 값이다. 같은 스크립트로 재야 한다.

subtask 런 진행 (wandb 값, 참고용):

| step | flow | subtask_ce | acc_first |
|---|---|---|---|
| 0 | 0.2958 | 6.3383 | 0.0000 |
| 1000 | 0.0511 | 0.0647 | 0.8825 |
| 1500 | 0.0354 | 0.0435 | 0.9075 |

**교란 요인**: subtask 런은 (a) CE 추가 (b) Task 지시문 제거, 두 가지가 동시에 다르다.
flow 에 차이가 나면 어느 쪽인지 이 비교로는 못 가른다. 차이가 없으면 둘 다 무해하다는
뜻이라 그대로 간다. 차이가 크면 `PI05_SUBTASK_W=0` + Task 제거만 한 런이 필요하다.

baseline 체크포인트 경로는 `checkpoints/<step>/...` 다 (`<step>/...` 가 아니다).
