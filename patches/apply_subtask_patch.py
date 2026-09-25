#!/usr/bin/env python3
"""pi0.5 에 subtask 문장 생성을 붙인다 (RSC_SUBTASK).

설계
  prompt 에 Subtask 슬롯을 만든다. 학습 때는 GT 문장을 teacher forcing 으로 넣고
  그 토큰 위치에만 CE 손실을 건다. 추론 때는 그 자리를 모델이 autoregressive 로
  생성한 뒤, 생성된 문장을 넣어 다시 prefix 를 돌리고 flow matching 을 한다.

  prompt 형식 (tokenizer.py)
    Task: {지시문}, State: {이산화 state}; Subtask: {문장};\\nAction:
                                           ^^^^^^^^^^^^^^^ 여기에만 CE

왜 이 방식인가 (토큰 1개짜리 head 대신)
  · VLM 자체가 조건화된다. suffix 에만 넣으면 action expert 만 본다
  · PaliGemma 의 언어 prior 를 쓴다. 새 embedding table 은 랜덤 초기화다
  · 나중에 10개 태스크를 합칠 때 "close the left gripper" 같은 구절이 공유된다
  · 디코드 비용은 KV cache 덕에 작다 (토큰당 1토큰 forward)

재사용
  logits 디코드·CE·decode loop 는 전부 pi0_fast.py 에 이미 있다. 새 파라미터는 없다
  (embedding tying: Embedder.decode 가 입력 임베딩 테이블을 전치해 쓴다).

scheduled sampling
  학습 내내 GT 를 넣으면 추론 때 자기 예측이 들어오는 순간 새 covariate shift 가
  생긴다. `RSC_SUBTASK_SS_START` 스텝부터 `RSC_SUBTASK_SS_MAX` 확률까지 선형으로
  올려, 그 확률로 GT 대신 **모델이 그 배치에서 예측한 문장**을 넣는다.
  기본은 0(비활성) — 먼저 teacher forcing 으로 baseline 과 비교한 뒤 켠다.

사용:  python3 apply_subtask_patch.py <repo 루트> [--check]
"""
from __future__ import annotations

import pathlib
import sys

MARK = "RSC_SUBTASK"

# ─────────────────────────────────────────────────────────────────────────
# 1. tokenizer — Subtask 슬롯 + 그 자리만 True 인 loss mask
# ─────────────────────────────────────────────────────────────────────────
TOK_FILE = "policy/pi05/src/openpi/models/tokenizer.py"

TOK_ANCHOR = '''    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)'''

TOK_NEW = '''    def tokenize(self, prompt: str, state: np.ndarray | None = None,
                 subtask: str | None = None) -> tuple[np.ndarray, ...]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            if subtask is not None:
                # {mark}: Subtask 슬롯. 앞부분(prefix)과 문장 부분을 따로 인코딩해
                # 문장 토큰 위치를 정확히 알아야 CE 를 거기에만 걸 수 있다.
                head = f"Task: {{cleaned_text}}, State: {{state_str}}; Subtask:"
                body = " " + subtask.strip()
                tail = ";\\nAction: "
                h = self._tokenizer.encode(head, add_bos=True)
                b = self._tokenizer.encode(body)
                t = self._tokenizer.encode(tail)
                tokens = h + b + t
                sub_span = (len(h), len(h) + len(b))
                return self._pad(tokens, sub_span)
            full_prompt = f"Task: {{cleaned_text}}, State: {{state_str}};\\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)'''.format(mark=MARK)

TOK_PAD_ANCHOR = '''        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:'''

TOK_PAD_NEW = '''        return np.asarray(tokens), np.asarray(mask)

    def _pad(self, tokens: list[int], sub_span: tuple[int, int]):
        """{mark}: 패딩 + subtask 토큰 구간만 True 인 loss mask 를 같이 낸다."""
        lo, hi = sub_span
        n = len(tokens)
        if n > self._max_len:
            logging.warning(
                f"Token length ({{n}}) exceeds max length ({{self._max_len}}), truncating. "
                "Subtask 문장이 잘릴 수 있으니 max_token_len 을 늘릴 것."
            )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
            n = self._max_len
        else:
            mask = [True] * n + [False] * (self._max_len - n)
            tokens = tokens + [0] * (self._max_len - n)
        loss = [lo <= i < min(hi, n) for i in range(self._max_len)]
        return np.asarray(tokens), np.asarray(mask), np.asarray(loss)


class FASTTokenizer:'''.format(mark=MARK)

# ─────────────────────────────────────────────────────────────────────────
# 2. transforms — TokenizePrompt 가 subtask 를 받아 넘기도록
# ─────────────────────────────────────────────────────────────────────────
TR_FILE = "policy/pi05/src/openpi/transforms.py"

TR_ANCHOR = '''class TokenizePrompt(DataTransformFn):'''
TR_NEW = '''@dataclasses.dataclass(frozen=True)
class InjectSubtask(DataTransformFn):
    """{mark}: frame_index -> GT subtask 문장.

    스케줄이 에피소드마다 동일하므로 프레임 번호만으로 결정된다
    (2026-09-25 검증: 10개 중 9개 태스크가 스케줄 길이와 정확히 일치).
    추론 때는 frame_index 가 없다 — 그때는 모델이 생성하므로 주입하지 않는다.
    """

    boundaries: tuple[int, ...] = ()      # 각 phase 의 끝 프레임
    sentences: tuple[str, ...] = ()

    def __call__(self, data: DataDict) -> DataDict:
        if "frame_index" not in data or not self.sentences:
            return data
        import numpy as _np
        f = int(_np.asarray(data["frame_index"]).reshape(-1)[0])
        i = int(_np.searchsorted(_np.asarray(self.boundaries), f, side="right"))
        i = min(i, len(self.sentences) - 1)
        return {{**data, "subtask": self.sentences[i], "phase_id": i}}


class TokenizePrompt(DataTransformFn):'''.format(mark=MARK)

TR_CALL_ANCHOR = '''        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}'''
TR_CALL_NEW = '''        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}'''

# ─────────────────────────────────────────────────────────────────────────

FILES = [TOK_FILE, TR_FILE]


def patch_one(root: pathlib.Path, rel: str, edits: list[tuple[str, str]], check: bool) -> bool:
    p = root / rel
    if not p.exists():
        print(f"  {rel}: 파일 없음")
        return False
    s = p.read_text()
    if MARK in s:
        print(f"  {rel}: 이미 적용됨")
        return True
    for anchor, _ in edits:
        if anchor not in s:
            print(f"  {rel}: ✗ 앵커를 못 찾음 — 업스트림이 바뀌었다\n      {anchor[:90]!r}")
            return False
    if check:
        print(f"  {rel}: 미적용 (적용 가능)")
        return False
    for anchor, new in edits:
        s = s.replace(anchor, new, 1)
    p.write_text(s)
    print(f"  {rel}: ✓ 적용")
    return True


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    check = "--check" in sys.argv
    print(f"{'확인' if check else '적용'}: {root}")
    ok = True
    ok &= patch_one(root, TOK_FILE, [(TOK_ANCHOR, TOK_NEW), (TOK_PAD_ANCHOR, TOK_PAD_NEW)], check)
    ok &= patch_one(root, TR_FILE, [(TR_ANCHOR, TR_NEW)], check)
    if check:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit("적용 실패 — 위 메시지 확인")
    print(f"\n1/3 단계 완료 (tokenizer + transform). 다음은 pi0.py 의 CE 손실과 디코드.")


if __name__ == "__main__":
    main()
