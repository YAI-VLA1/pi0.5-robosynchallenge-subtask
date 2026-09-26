#!/usr/bin/env python3
"""추론 시 subtask 문장을 생성하는 경로 (RSC_DECODE). 멱등, --check 지원.

학습에는 필요 없다 (teacher forcing). 평가 전에 반드시 적용해야 한다.

무엇을 하나
  1. TokenizePrompt 가 추론에서도 **빈 슬롯**을 만든다.
     학습에서는 InjectSubtask 가 GT 문장을 넣지만(repack 그룹 = 학습 전용),
     추론에는 아무도 안 넣는다. 슬롯이 없으면 생성할 자리가 없고,
     token_loss_mask 가 없어 attention 이 causal 로 바뀌지도 않는다.
  2. sample_actions 앞에서 슬롯 14자리를 greedy 로 채운다.

비용
  LLM forward 가 14번 늘어난다. 이미지 토큰(968 중 768)은 한 번만 계산해
  SigLIP 재계산은 피했지만, LLM 은 매 스텝 전체를 다시 돈다.
  KV cache 로 증분 디코드하면 1/14 로 줄일 수 있다 — 지금 구조에서 유효하다
  (슬롯 앞 구간은 cumsum 0 이라 슬롯과 무관하게 캐시할 수 있다).
  먼저 맞게 돌리고, 평가 시간이 문제가 되면 그때 최적화한다.

한계
  greedy 만 구현했다. 문장이 15개 고정 집합이라 sampling 이 득이 될 이유가 없다.
"""
from __future__ import annotations

import pathlib
import sys

MARK = "RSC_DECODE"

# ── 1. transforms.py — 추론에서도 빈 슬롯을 만든다 ──────────────────────
TR = "policy/pi05/src/openpi/transforms.py"
TR_A = '''        # RSC_SUBTASK: subtask 가 있으면 슬롯을 끼우고 그 자리만 True 인 loss mask 를 같이 낸다.
        subtask = data.pop("subtask", None)
        if subtask is not None and state is not None:'''
TR_B = '''        # RSC_SUBTASK: subtask 가 있으면 슬롯을 끼우고 그 자리만 True 인 loss mask 를 같이 낸다.
        subtask = data.pop("subtask", None)
        # {mark}: 추론에는 InjectSubtask 가 돌지 않는다(repack 그룹 = 학습 전용).
        # 빈 슬롯이라도 만들어야 (a) 생성할 자리가 생기고 (b) token_loss_mask 가 생겨
        # embed_prefix 가 그 구간을 causal 로 잡는다.
        if subtask is None and self.subtask_slot and state is not None:
            subtask = ""
        if subtask is not None and state is not None:'''.format(mark=MARK)

TR_FIELD_A = '''class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False'''
TR_FIELD_B = '''class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
    # {mark}: True 면 subtask 가 안 들어와도 빈 슬롯을 만든다 (추론 경로).
    subtask_slot: bool = False'''.format(mark=MARK)

# ── 2. tokenizer.py — 빈 문장도 슬롯을 만들게 ───────────────────────────
TOK = "policy/pi05/src/openpi/models/tokenizer.py"
TOK_A = '''        body = self._tokenizer.encode(" " + subtask.strip())'''
TOK_B = '''        # {mark}: 빈 문자열이면 슬롯 전체가 패딩이다 (추론에서 모델이 채운다).
        body = self._tokenizer.encode(" " + subtask.strip()) if subtask.strip() else []'''.format(mark=MARK)

# ── 3. pi0.py — sample_actions 앞에 greedy 디코드 ───────────────────────
PI0 = "policy/pi05/src/openpi/models/pi0.py"
PI0_A = '''        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps'''
PI0_B = '''        observation = _model.preprocess_observation(None, observation, train=False)

        # {mark}: subtask 슬롯을 모델이 채운다. 학습 때는 GT 를 넣었지만(teacher forcing)
        # 추론에는 없다. 채운 뒤의 prompt 로 flow matching 을 돌린다.
        if _RSC_SUBTASK_W > 0.0 and observation.token_loss_mask is not None:
            observation = self._rsc_generate_subtask(observation)

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps'''.format(mark=MARK)

PI0_METHOD_A = '''    @override
    def sample_actions('''
PI0_METHOD_B = '''    def _rsc_generate_subtask(self, obs):
        """{mark}: subtask 슬롯 {slot}자리를 greedy 로 채운 observation 을 돌려준다.

        슬롯 앞 구간(이미지·Task·State·"Subtask:")은 cumsum 0 이라 슬롯과 무관하고,
        슬롯 k 번째는 0..k 만 본다 (embed_prefix 의 causal 블록). 따라서 앞에서부터
        한 자리씩 확정해 나가는 것이 옳다.

        이미지 토큰은 한 번만 계산한다. LLM 은 매 스텝 전체를 다시 도는데,
        KV cache 로 줄일 수 있다 — 필요해지면 그때 한다.
        """
        tok = obs.tokenized_prompt
        lm = obs.token_loss_mask
        start = jnp.argmax(lm.astype(jnp.int32), axis=-1)          # (b,)

        # 이미지 부분은 슬롯 내용과 무관하므로 한 번만 만든다.
        tokens0, input_mask, ar_mask = self.embed_prefix(obs)
        n_img = tokens0.shape[1] - tok.shape[1]
        img_part = tokens0[:, :n_img]
        attn = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        def one(tok, k):
            text = self.PaliGemma.llm(tok, method="embed")
            full = jnp.concatenate([img_part, text], axis=1)
            (out, _), _ = self.PaliGemma.llm([full, None], mask=attn, positions=positions)
            # 슬롯 k 번째를 맞히는 자리는 (n_img + start + k - 1)
            at_idx = (n_img + start + k - 1)[:, None, None]
            h = jnp.take_along_axis(out, at_idx, axis=1)           # (b, 1, d)
            nxt = jnp.argmax(self.PaliGemma.llm(h, method="decode"), axis=-1)  # (b, 1)
            return jnp.put_along_axis(tok, (start + k)[:, None], nxt, axis=1, inplace=False)

        for k in range(_RSC_SUBTASK_SLOT):
            tok = one(tok, k)
        return dataclasses.replace(obs, tokenized_prompt=tok)

    @override
    def sample_actions('''.format(mark=MARK, slot="{slot}").replace("{slot}", "_RSC_SUBTASK_SLOT")

PI0_IMP_A = "import jax.numpy as jnp"
PI0_IMP_B = "import dataclasses\nimport jax.numpy as jnp"

# ── 4. config.py — 추론 경로에 슬롯 플래그 ──────────────────────────────
CFG = "policy/pi05/src/openpi/training/config.py"
CFG_A = '''                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),'''
CFG_B = '''                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                            # {mark}: 환경변수로 켠다. 학습·추론 양쪽에 같은 값이 들어가야 한다.
                            subtask_slot=bool(float(_os.environ.get("PI05_SUBTASK_W", "0")) > 0),
                        ),'''.format(mark=MARK)

EDITS = [
    (TOK, [(TOK_A, TOK_B)]),
    (TR, [(TR_FIELD_A, TR_FIELD_B), (TR_A, TR_B)]),
    (PI0, [(PI0_IMP_A, PI0_IMP_B), (PI0_METHOD_A, PI0_METHOD_B), (PI0_A, PI0_B)]),
    (CFG, [(CFG_A, CFG_B)]),
]


def patch_one(root: pathlib.Path, rel: str, edits, check: bool) -> bool:
    p = root / rel
    if not p.exists():
        print(f"  {rel}: 파일 없음")
        return False
    s = p.read_text()
    if MARK in s:
        print(f"  {rel}: 이미 적용됨")
        return True
    for a, _ in edits:
        n = s.count(a)
        if n != 1:
            print(f"  {rel}: ✗ 앵커가 {n}곳 — 더 좁혀야 한다\n      {a.splitlines()[0][:88]!r}")
            return False
    if check:
        print(f"  {rel}: 미적용 (적용 가능)")
        return False
    for a, b in edits:
        s = s.replace(a, b, 1)
    if rel == CFG and "import os as _os" not in s:
        s = s.replace("import dataclasses", "import os as _os\nimport dataclasses", 1)
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
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
