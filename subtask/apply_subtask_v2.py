#!/usr/bin/env python3
"""apply_subtask_patch.py 이후의 2차 수정 (RSC_SUBTASK_V2). 멱등, --check 지원.

왜 별도 파일인가
  이 네 가지는 2026-09-25 반복 수정 중 **라이브 파일에만** 손으로 넣고 패치
  스크립트에 반영하지 않았다. 다음날 파드가 죽어 새 클론을 만들었을 때 전부
  빠졌고, 프롬프트 형식이 달라져 재개 loss 가 0.015 -> 10.25 로 튀었다.
  큰 템플릿 문자열을 다시 쓰는 것보다 델타를 따로 두는 쪽이 검증하기 쉽다.

  교훈: 라이브 파일을 손으로 고쳤으면 그 자리에서 패치 스크립트에 반영할 것.

내용
  1. 원본 Task 지시문 제거 — 에피소드 내내 고정이라 정보가 0 인데 22토큰을 먹는다
  2. EOS 종료 — 문장 끝을 EOS 로 찍고 그 뒤 패딩은 CE 에서 뺀다 (손실의 40% 였다)
  3. subtask_acc / subtask_acc_first — logits 가 이미 있으니 거의 공짜다.
     teacher forcing 에서 뒤쪽 단어는 쉬워서 **첫 토큰 정확도만이** 의미 있다
  4. return_parts + train.py has_aux — flow 와 CE 를 wandb 에 따로 올린다

사용:  python3 apply_subtask_v2.py <repo 루트> [--check]
"""
from __future__ import annotations

import pathlib
import sys

MARK = "RSC_SUBTASK_V2"

TOK = "policy/pi05/src/openpi/models/tokenizer.py"
PI0 = "policy/pi05/src/openpi/models/pi0.py"
TRAIN = "policy/pi05/scripts/train.py"

# ── 1. Task 지시문 제거 ────────────────────────────────────────────────
T1_A = '''        head = self._tokenizer.encode(
            f"Task: {cleaned_text}, State: {state_str}; Subtask:", add_bos=True
        )'''
T1_B = '''        # {mark}: 원본 Task 지시문을 뺀다. 에피소드 내내 고정이라 정보가 0 인데
        # 22토큰을 먹고, subtask 가 유일한 언어 신호가 되도록 하는 편이 조건화가 선명하다.
        # State 가 Subtask 앞에 오는 것이 중요하다 — 슬롯이 causal 이라 생성 시점에
        # state 를 이미 봐야 한다. (다태스크로 갈 때는 Task 를 되살려야 한다.)
        _ = cleaned_text
        head = self._tokenizer.encode(f"State: {{state_str}}; Subtask:", add_bos=True)'''.format(mark=MARK)

# ── 2. EOS 종료 ────────────────────────────────────────────────────────
T2_A = '''        if len(body) > self.SUBTASK_SLOT:
            logging.warning(f"subtask 문장이 슬롯({self.SUBTASK_SLOT})보다 길다: {len(body)} — 자른다")
            body = body[: self.SUBTASK_SLOT]
        n_body = len(body)
        pad_id = self._tokenizer.pad_id() if self._tokenizer.pad_id() >= 0 else 0
        body = body + [pad_id] * (self.SUBTASK_SLOT - n_body)'''
T2_B = '''        # {mark}: 문장 끝을 EOS 로 찍는다. 슬롯은 jit 용 고정 버퍼일 뿐이고
        # 생성 길이는 EOS 가 정한다. 패딩까지 CE 를 걸면 손실의 40% 가
        # '0 을 맞혀라' 가 된다 (문장+EOS 5~13토큰, 슬롯 14 -> 패딩 39.5%).
        eos_id = self._tokenizer.eos_id()
        pad_id = self._tokenizer.pad_id() if self._tokenizer.pad_id() >= 0 else 0
        body = body + [eos_id]
        if len(body) > self.SUBTASK_SLOT:
            logging.warning(f"문장+EOS 가 슬롯({{self.SUBTASK_SLOT}})을 넘는다: {{len(body)}}")
            body = body[: self.SUBTASK_SLOT - 1] + [eos_id]
        n_body = len(body)
        body = body + [pad_id] * (self.SUBTASK_SLOT - n_body)'''.format(mark=MARK)

# ── 3. EOS 뒤 손실 제외 + 정확도 지표 ──────────────────────────────────
P1_A = '''        msk = jnp.take_along_axis(lm, idx, axis=1).astype(jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, tgt)
        return (ce * msk).sum(-1) / jnp.maximum(msk.sum(-1), 1.0)'''
P1_B = '''        msk = jnp.take_along_axis(lm, idx, axis=1).astype(jnp.float32)
        # {mark}: EOS **까지만** 학습한다. 그 뒤 패딩은 자명해서 손실을 희석시킬 뿐이다.
        # token_loss_mask 는 슬롯 전체를 덮는다 — attention 블록 경계로도 쓰이므로
        # 거기서 줄이면 causal 구조가 깨진다. 그래서 여기서 따로 자른다.
        eos = (tgt == _RSC_EOS_ID).astype(jnp.int32)
        after_eos = jnp.cumsum(eos, axis=-1) - eos          # EOS 다음 자리부터 1
        msk = msk * (after_eos == 0).astype(jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, tgt)
        per_sample = (ce * msk).sum(-1) / jnp.maximum(msk.sum(-1), 1.0)

        # {mark}: 진단 지표. logits 를 이미 갖고 있으니 거의 공짜다.
        #   acc       — EOS 까지의 평균 (teacher forcing 이라 뒤로 갈수록 쉽다)
        #   acc_first — 슬롯 첫 자리. 여기서 phase 판단이 일어나고 나머지는 따라온다.
        #               이게 진짜 "장면을 보고 subtask 를 아는가" 지표다.
        hit = (jnp.argmax(logits, axis=-1) == tgt).astype(jnp.float32)
        acc = jnp.sum(hit * msk) / jnp.maximum(jnp.sum(msk), 1.0)
        acc_first = jnp.mean(hit[:, 0])
        return per_sample, acc, acc_first'''.format(mark=MARK)

P2_A = '_RSC_SUBTASK_SLOT = 14'
P2_B = '_RSC_SUBTASK_SLOT = 14\n_RSC_EOS_ID = 1          # PaliGemma sentencepiece eos_id'

# ── 4. return_parts ────────────────────────────────────────────────────
P3_A = '''    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:'''
P3_B = '''    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *,
        train: bool = False, return_parts: bool = False
    ):'''

P4_A = '''        if _RSC_SUBTASK_W <= 0.0 or observation.token_loss_mask is None:
            return flow
        ce = self._rsc_subtask_ce(observation, prefix_out)
        return flow + _RSC_SUBTASK_W * ce[:, None]'''
P4_B = '''        if _RSC_SUBTASK_W <= 0.0 or observation.token_loss_mask is None:
            if return_parts:
                return flow, {{"flow": jnp.mean(flow), "subtask_ce": jnp.zeros(())}}
            return flow
        ce, acc, acc_first = self._rsc_subtask_ce(observation, prefix_out)
        total = flow + _RSC_SUBTASK_W * ce[:, None]
        if return_parts:
            # {mark}: 합친 loss 만 보면 flow 가 나빠지는지 CE 가 안 내려가는지 가릴 수 없다.
            return total, {{"flow": jnp.mean(flow), "subtask_ce": jnp.mean(ce),
                           "subtask_acc": acc, "subtask_acc_first": acc_first}}
        return total'''.format(mark=MARK)

# ── 5. train.py has_aux ────────────────────────────────────────────────
R1_A = '''        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)'''
R1_B = '''        chunked_loss, parts = model.compute_loss(rng, observation, actions, train=True, return_parts=True)
        return jnp.mean(chunked_loss), parts'''

R2_A = '''    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)'''
R2_B = '''    (loss, loss_parts), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )'''

R3_A = '''    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }'''
R3_B = '''    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        # RSC_SUBTASK_V2: flow / CE / 정확도를 따로 올린다.
        **{f"loss/{k}": v for k, v in (loss_parts or {}).items()},
    }'''

EDITS = [
    (TOK, [(T1_A, T1_B), (T2_A, T2_B)]),
    (PI0, [(P2_A, P2_B), (P1_A, P1_B), (P3_A, P3_B), (P4_A, P4_B)]),
    (TRAIN, [(R1_A, R1_B), (R2_A, R2_B), (R3_A, R3_B)]),
]


def patch_one(root: pathlib.Path, rel: str, edits, check: bool) -> bool:
    p = root / rel
    if not p.exists():
        print(f"  {rel}: 파일 없음")
        return False
    s = p.read_text()
    if MARK in s or (rel == TRAIN and "has_aux" in s):
        print(f"  {rel}: 이미 적용됨")
        return True
    for a, _ in edits:
        n = s.count(a)
        if n != 1:
            print(f"  {rel}: ✗ 앵커가 {n}곳 — {a.splitlines()[0][:80]!r}")
            return False
    if check:
        print(f"  {rel}: 미적용 (적용 가능)")
        return False
    for a, b in edits:
        s = s.replace(a, b, 1)
    p.write_text(s)
    print(f"  {rel}: ✓ 적용")
    return True


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    check = "--check" in sys.argv
    print(f"{'확인' if check else '적용'}: {root}")
    ok = True
    for rel, edits in EDITS:
        ok &= patch_one(root, rel, edits, check)
    if check:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit("적용 실패 — 위 메시지 확인")
    print("\n완료. 프롬프트는 `State: ...; Subtask: <14>;\\nAction: ` 이고")
    print("wandb 에 loss/flow, loss/subtask_ce, loss/subtask_acc(_first) 가 올라간다.")


if __name__ == "__main__":
    main()
