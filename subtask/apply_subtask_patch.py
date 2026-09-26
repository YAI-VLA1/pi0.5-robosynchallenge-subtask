#!/usr/bin/env python3
"""pi0.5 에 subtask 문장 생성을 붙인다 (RSC_SUBTASK). 멱등, --check 지원.

설계
  prompt 에 **고정 길이 14토큰** Subtask 슬롯을 만든다.
    Task: {지시문}, State: {이산화 state}; Subtask: {문장 14토큰};\\nAction:
                                           ^^^^^^^^^^^^^^^^ 여기에만 CE
  학습: GT 문장을 teacher forcing 으로 넣고 그 자리에만 CE 를 건다.
  추론: 그 자리를 모델이 14스텝 생성한 뒤 flow matching (별도 패치, 학습에는 불필요).

왜 고정 길이인가
  · 디코드 스텝 수가 상수라 unroll 로 jit 이 된다 (KVCache 가 concat 방식이라
    가변 길이면 트레이스가 매번 달라진다)
  · 문장 최장이 12토큰이라 14면 여유가 있다

왜 gemma_fast 를 안 쓰나 (2026-09-25 정정)
  pi0_fast 는 `gemma_fast.py` 라는 **별도 모듈**을 쓴다. pi0.5 가 쓰는 `gemma.py` 에는
  `return_prelogits` 도 `decode` 도 없고 호출 시그니처도 다르다. 복사가 아니라
  `Module.decode` 를 3줄 추가하는 쪽이 맞다. logit head 는 embedding tying
  (`Embedder.decode` 가 입력 임베딩 테이블을 전치)이라 **새 파라미터가 없다**.

CE 위치 계산
  prefix 는 [이미지 토큰 | 텍스트 토큰] 순이고 텍스트가 뒤에 온다.
  n_img = prefix_out.shape[1] - tokenized_prompt.shape[1].
  다음 토큰 예측이므로 텍스트 i 를 맞히는 자리는 prefix 위치 (n_img + i - 1).
  subtask 구간의 시작은 `argmax(token_loss_mask)` 로 샘플마다 구한다
  (state 문자열 길이가 가변이라 오프셋이 고정이 아니다).
  14자리만 gather 하므로 vocab 매트멀이 14 x 257k 로 끝난다.
"""
from __future__ import annotations

import pathlib
import sys

MARK = "RSC_SUBTASK"
SLOT = 14   # subtask 슬롯 토큰 수 (문장 최장 12 + 여유)

# ═════════════════════════════════════════════════════════════════════════
# 1. gemma.py — Module 에 decode 노출 (logit head, 새 파라미터 없음)
# ═════════════════════════════════════════════════════════════════════════
GEMMA = "policy/pi05/src/openpi/models/gemma.py"
GEMMA_A = '''    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)
'''
GEMMA_B = '''    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    # {mark}: hidden state -> vocab logits. Embedder.decode 가 입력 임베딩 테이블을
    # 전치해 쓰므로(embedding tying) 새 파라미터가 없다.
    def decode(self, x):
        return self.embedder.decode(x)
'''.format(mark=MARK)

# ═════════════════════════════════════════════════════════════════════════
# 2. tokenizer.py — 고정 길이 Subtask 슬롯 + 그 자리만 True 인 loss mask
# ═════════════════════════════════════════════════════════════════════════
TOK = "policy/pi05/src/openpi/models/tokenizer.py"
TOK_A = '''    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\\n", " ")
        if state is not None:'''
TOK_B = '''    # {mark}
    SUBTASK_SLOT = {slot}

    def tokenize_with_subtask(self, prompt: str, state: np.ndarray, subtask: str):
        """{mark}: Subtask 슬롯을 고정 길이로 끼운다.

        반환 (tokens, mask, loss_mask). loss_mask 는 subtask 슬롯 중 **실제 문장
        토큰 자리**만 True 다 (패딩 자리는 False — 패딩을 맞히라고 가르치지 않는다).
        """
        cleaned_text = prompt.strip().replace("_", " ").replace("\\n", " ")
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        head = self._tokenizer.encode(
            f"Task: {{cleaned_text}}, State: {{state_str}}; Subtask:", add_bos=True
        )
        body = self._tokenizer.encode(" " + subtask.strip())
        if len(body) > self.SUBTASK_SLOT:
            logging.warning(f"subtask 문장이 슬롯({{self.SUBTASK_SLOT}})보다 길다: {{len(body)}} — 자른다")
            body = body[: self.SUBTASK_SLOT]
        n_body = len(body)
        pad_id = self._tokenizer.pad_id() if self._tokenizer.pad_id() >= 0 else 0
        body = body + [pad_id] * (self.SUBTASK_SLOT - n_body)
        tail = self._tokenizer.encode(";\\nAction: ")

        tokens = head + body + tail
        lo = len(head)
        n = len(tokens)
        if n > self._max_len:
            raise ValueError(
                f"subtask 를 넣은 prompt 가 max_token_len 을 넘는다 ({{n}} > {{self._max_len}})."
            )
        mask = [True] * n + [False] * (self._max_len - n)
        tokens = tokens + [pad_id] * (self._max_len - n)
        # {mark}: 슬롯 전체가 True 다. (a) 패딩까지 생성하도록 가르쳐야 추론 디코드가
        # 항상 정확히 {slot}스텝이고, (b) 이 마스크가 attention causal 블록 경계로도 쓰인다.
        loss = [lo <= i < lo + self.SUBTASK_SLOT for i in range(self._max_len)]
        return np.asarray(tokens), np.asarray(mask), np.asarray(loss)

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\\n", " ")
        if state is not None:'''.format(mark=MARK, slot=SLOT)

# ═════════════════════════════════════════════════════════════════════════
# 3. transforms.py — InjectSubtask + TokenizePrompt 의 subtask 경로
# ═════════════════════════════════════════════════════════════════════════
TR = "policy/pi05/src/openpi/transforms.py"
TR_A = '''class TokenizePrompt(DataTransformFn):'''
TR_B = '''class InjectSubtask(DataTransformFn):
    """{mark}: frame_index -> GT subtask 문장.

    sim 에피소드는 스크립트 재생이라 구간 경계가 에피소드마다 같다
    (2026-09-25 검증: 10개 중 9개 태스크가 스케줄 길이와 정확히 일치).
    따라서 프레임 번호만으로 결정된다.

    추론 때는 frame_index 가 없다 — 그때는 모델이 생성하므로 아무것도 넣지 않는다.
    """

    ends: tuple[int, ...] = ()          # 각 phase 의 끝 프레임 (오름차순)
    sentences: tuple[str, ...] = ()

    def __call__(self, data: DataDict) -> DataDict:
        if not self.sentences or "frame_index" not in data:
            return data
        f = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        i = min(int(np.searchsorted(np.asarray(self.ends), f, side="right")),
                len(self.sentences) - 1)
        return {{**data, "subtask": self.sentences[i]}}


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):'''.format(mark=MARK)

TR_C = '''        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}'''
TR_D = '''        # {mark}: subtask 가 있으면 슬롯을 끼우고 그 자리만 True 인 loss mask 를 같이 낸다.
        subtask = data.pop("subtask", None)
        if subtask is not None and state is not None:
            if not isinstance(subtask, str):
                subtask = subtask.item()
            tokens, token_masks, loss_mask = self.tokenizer.tokenize_with_subtask(prompt, state, subtask)
            return {{**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks,
                    "token_loss_mask": loss_mask}}

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {{**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}}'''.format(mark=MARK)

# ═════════════════════════════════════════════════════════════════════════
# 4. libero_policy.py — subtask 를 통과시킨다 (여기서 dict 를 새로 만든다)
# ═════════════════════════════════════════════════════════════════════════
POL = "policy/pi05/src/openpi/policies/libero_policy.py"
POL_A = '''        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs'''
POL_B = '''        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # {mark}: 이 dict 는 새로 만들어지므로 명시하지 않으면 subtask 가 사라진다.
        if "subtask" in data:
            inputs["subtask"] = data["subtask"]

        return inputs'''.format(mark=MARK)

# ═════════════════════════════════════════════════════════════════════════
# 5. pi0.py — CE 손실
# ═════════════════════════════════════════════════════════════════════════
PI0 = "policy/pi05/src/openpi/models/pi0.py"
PI0_A = '''        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # RSC_LOSS_MASK: 패딩 축을 손실에서 제외한다. pad_to_dim 이 뒤에 붙이므로 앞 d 축만 쓴다.
        d = _RSC_REAL_ACTION_DIM
        if 0 < d < v_t.shape[-1]:
            return jnp.mean(jnp.square(v_t[..., :d] - u_t[..., :d]), axis=-1)
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)'''
PI0_B = '''        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # RSC_LOSS_MASK: 패딩 축을 손실에서 제외한다. pad_to_dim 이 뒤에 붙이므로 앞 d 축만 쓴다.
        d = _RSC_REAL_ACTION_DIM
        if 0 < d < v_t.shape[-1]:
            flow = jnp.mean(jnp.square(v_t[..., :d] - u_t[..., :d]), axis=-1)
        else:
            flow = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # {mark}: subtask 토큰에만 CE. token_loss_mask 가 없으면 아무것도 하지 않는다.
        if _RSC_SUBTASK_W <= 0.0 or observation.token_loss_mask is None:
            return flow
        ce = self._rsc_subtask_ce(observation, prefix_out)
        return flow + _RSC_SUBTASK_W * ce[:, None]

    def _rsc_subtask_ce(self, observation, prefix_out):
        """{mark}: subtask 슬롯 {slot}자리만 gather 해서 CE 를 낸다.

        prefix 는 [이미지 | 텍스트] 순이라 텍스트 i 를 맞히는 자리는 (n_img + i - 1) 이다.
        슬롯 시작은 샘플마다 다르므로(state 문자열 길이가 가변) argmax 로 구한다.
        """
        tok = observation.tokenized_prompt
        lm = observation.token_loss_mask
        n_img = prefix_out.shape[1] - tok.shape[1]
        start = jnp.argmax(lm.astype(jnp.int32), axis=-1)               # (b,)
        idx = start[:, None] + jnp.arange(_RSC_SUBTASK_SLOT)            # (b, L)
        idx = jnp.clip(idx, 0, tok.shape[1] - 1)
        pre = jnp.take_along_axis(prefix_out, (n_img + idx - 1)[..., None], axis=1)
        logits = self.PaliGemma.llm(pre, method="decode").astype(jnp.float32)
        tgt = jnp.take_along_axis(tok, idx, axis=1)
        msk = jnp.take_along_axis(lm, idx, axis=1).astype(jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, tgt)
        return (ce * msk).sum(-1) / jnp.maximum(msk.sum(-1), 1.0)'''.format(mark=MARK, slot=SLOT)

PI0_PRE_A = '''        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
'''
PI0_PRE_B = '''        ar_mask = jnp.array(ar_mask)

        # {mark}: 기본 prefix 는 **완전 양방향**이다(위 ar_mask 가 전부 False).
        # 그대로 두면 위치 p 의 hidden state 가 이미 토큰 p+1 을 보므로 다음 토큰 예측
        # CE 가 '복사'로 풀린다. 실제로 그렇게 됐다 — step 0 의 7.29 가 800스텝 만에
        # 0.046 으로 무너졌다 (2026-09-25).
        #
        # subtask 슬롯만 causal 블록으로 만든다. make_attn_mask 는 cumsum(ar_mask) 로
        # 블록을 나누므로:
        #   · 이미지/Task/State (cumsum 0) 는 슬롯을 못 본다 — 맞다, 생성 대상이다
        #   · 슬롯 k 번째는 0..k 만 본다 — causal
        #   · 뒤의 ";\\nAction: " 은 슬롯 전체를 본다 — 조건화에 필요하다
        if obs.token_loss_mask is not None and obs.tokenized_prompt is not None:
            n_img = tokens.shape[1] - obs.tokenized_prompt.shape[1]
            ar_mask = jnp.concatenate(
                [jnp.zeros((obs.token_loss_mask.shape[0], n_img), dtype=bool),
                 obs.token_loss_mask.astype(bool)],
                axis=1,
            )
        return tokens, input_mask, ar_mask
'''.format(mark=MARK)

# @at.typecheck 이 1-D 를 강제하므로 반환형 주석을 넓힌다
PI0_SIG_A = '    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:'
PI0_SIG_B = '    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Array]:'

PI0_AR_A = '        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)'
PI0_AR_B = '        # {mark}: prefix_ar_mask 가 샘플별 2-D 가 되었으므로 suffix 도 맞춰 붙인다.\n        # (suffix 는 cumsum 이 prefix 최대값보다 크므로 prefix 전체를 볼 수 있다 — 의도한 대로다.)\n        if prefix_ar_mask.ndim == 2:\n            ar_mask = jnp.concatenate(\n                [prefix_ar_mask,\n                 jnp.broadcast_to(suffix_ar_mask, (prefix_ar_mask.shape[0], suffix_ar_mask.shape[-1]))],\n                axis=1,\n            )\n        else:\n            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)'.format(mark=MARK)

PI0_HDR_A = "class Pi0(_model.BaseModel):"
PI0_HDR_B = '''# {mark}: subtask CE 손실 가중치. 0 이면 완전히 꺼진다(원래 동작).
_RSC_SUBTASK_W = float(_os.environ.get("PI05_SUBTASK_W", "0.0"))
_RSC_SUBTASK_SLOT = {slot}
logging.info("{mark}: CE 가중치 = %s, 슬롯 = %d", _RSC_SUBTASK_W, _RSC_SUBTASK_SLOT)


class Pi0(_model.BaseModel):'''.format(mark=MARK, slot=SLOT)

EDITS = [
    (GEMMA, [(GEMMA_A, GEMMA_B)]),
    (TOK, [(TOK_A, TOK_B)]),
    (TR, [(TR_A, TR_B), (TR_C, TR_D)]),
    (POL, [(POL_A, POL_B)]),
    (PI0, [(PI0_HDR_A, PI0_HDR_B), (PI0_A, PI0_B),
           (PI0_SIG_A, PI0_SIG_B), (PI0_PRE_A, PI0_PRE_B),
           (PI0_AR_A, PI0_AR_B)]),
]


def patch_one(root: pathlib.Path, rel: str, edits, check: bool, scope: str | None = None) -> bool:
    p = root / rel
    if not p.exists():
        print(f"  {rel}: 파일 없음")
        return False
    s = p.read_text()
    if MARK in s:
        print(f"  {rel}: 이미 적용됨")
        return True
    # scope 가 주어지면 그 클래스 본문 안에서만 찾는다. libero_policy.py 의
    # `if "prompt" in data:` 는 LiberoInputs 와 EmbodiChainInputs 양쪽에 똑같이 있다.
    lo, hi = 0, len(s)
    if scope is not None:
        if scope not in s:
            print(f"  {rel}: ✗ 클래스 {scope} 를 못 찾음")
            return False
        lo = s.index(scope)
        nxt = s.find("\nclass ", lo + 1)
        hi = nxt if nxt != -1 else len(s)
    body = s[lo:hi]
    for a, _ in edits:
        # 앵커가 여러 곳에 있으면 replace(...,1) 이 엉뚱한 클래스를 잡는다.
        # config.py 와 libero_policy.py 에서 실제로 그렇게 당했다 (2026-09-25).
        if body.count(a) > 1:
            print(f"  {rel}: ✗ 앵커가 {body.count(a)}곳에 있다 — 더 좁혀야 한다\n      {a.splitlines()[0][:90]!r}")
            return False
        if a not in body:
            print(f"  {rel}: ✗ 앵커 없음 — 업스트림이 바뀌었다\n      {a.splitlines()[0][:90]!r}")
            return False
    if check:
        print(f"  {rel}: 미적용 (적용 가능)")
        return False
    for a, b in edits:
        body = body.replace(a, b, 1)
    s = s[:lo] + body + s[hi:]
    if rel == PI0 and "import optax" not in s:
        s = s.replace("import jax.numpy as jnp", "import jax.numpy as jnp\nimport optax", 1)
    p.write_text(s)
    print(f"  {rel}: ✓ 적용")
    return True


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    check = "--check" in sys.argv
    print(f"{'확인' if check else '적용'}: {root}")
    ok = True
    scopes = {POL: "class EmbodiChainInputs"}
    for rel, edits in EDITS:
        ok &= patch_one(root, rel, edits, check, scopes.get(rel))
    if check:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit("적용 실패 — 위 메시지 확인")
    print(f"\n완료. CE 는 기본 꺼져 있다 — 학습 시 PI05_SUBTASK_W=1.0 을 줄 것.")
    print("데이터 쪽(frame_index 통과 + InjectSubtask 배선)은 prepare_subtask_config.py 가 한다.")


if __name__ == "__main__":
    main()
