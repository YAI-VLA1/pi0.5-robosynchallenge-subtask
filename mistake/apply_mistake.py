#!/usr/bin/env python3
"""pi0.7 식 Mistake 메타데이터를 프롬프트에 배선한다.

프롬프트 형식
  baseline  Task: {task}, State: {s}, Mistake: false;\\nAction:
  subtask   State: {s}; Mistake: false; Subtask: {slot};\\nAction:
  필드가 드롭되면(학습 중 5%) 그 조각이 통째로 빠져 **기존 형식과 바이트 단위로 같다**.
  81999 에서 이어붙일 때 모델이 처음부터 낯선 프롬프트만 보지 않게 하는 장치다.

경로
  InjectMistake(repack, 학습 전용)  episode_index+frame_index -> data["mistake"]
    True / False / None(=드롭, 필드 생략)
  EmbodiChainInputs                 dict 를 새로 만들므로 명시 통과가 필요하다
  TokenizePrompt(학습+추론 공용)     키가 없으면(추론) mistake_field 일 때 False 로 둔다

켜기: PI05_MISTAKE=1 (학습·추론 양쪽에 같은 값을 줘야 한다)
"""
import pathlib, sys

MARK = "RSC_MISTAKE"


def _norm(l):
    """줄 끝의 `,` `)` `:` 와 공백을 떼고 비교한다.

    패치끼리 **같은 줄**(함수 시그니처 등)에 인자를 덧붙이면, 먼저 적용된 패치가
    넣은 줄의 끝이 `= None):` 에서 `= None,` 로 바뀐다. 통째로 비교하면
    '아직 적용 안 됨' 으로 오판하고 사라진 앵커를 찾다 죽는다. (2026-10-05)
    """
    return l.strip().rstrip(",):").strip()


def scoped_replace(text, old, new, *, scope=None, path=""):
    """앵커 유일성을 검사하고 바꾼다. scope 가 있으면 그 클래스 본문 안에서만 찾는다."""
    # 멱등 판정: new 에만 있고 old 에는 없는 줄들이 이미 전부 들어 있으면 적용됐다.
    # `new in text` 로는 안 된다 — 다른 패치가 사이에 줄을 끼우면 통째 비교가 깨져
    # 같은 블록을 두 번 넣는다 (2026-10-04 libero_policy.py 에서 실제로 그랬다).
    oldn = {_norm(l) for l in old.splitlines()}
    # 주석은 판정에서 뺀다 — 패치마다 MARK 문자열이 달라 같은 코드를 '미적용' 으로 본다.
    added = [_norm(l) for l in new.splitlines()
             if l.strip() and not l.strip().startswith("#") and _norm(l) not in oldn]
    hay = text
    if scope and scope in text:
        i = text.index(scope)
        j = text.find("\nclass ", i + 1)
        hay = text[i: len(text) if j < 0 else j]
    hayn = [_norm(l) for l in hay.splitlines()]
    # 정확히 같은 줄이 아니라 **부분 문자열** 로 본다. 패치끼리 같은 호출에
    # 인자를 덧붙이면 `f(a, b)` 가 `f(a, b, c)` 가 되어 완전 일치가 깨진다.
    def _has(ln):
        return any(ln and ln in h for h in hayn)
    if added and all(_has(ln) for ln in added):
        return text, False
    body, lo = text, 0
    if scope:
        i = text.index(scope)
        j = text.find("\nclass ", i + 1)
        j = len(text) if j < 0 else j
        body, lo = text[i:j], i
    n = body.count(old)
    if n != 1:
        raise SystemExit(f"★ {path}: 앵커가 {n} 번 나온다 (1 이어야 함)\n{old[:120]}")
    k = lo + body.index(old)
    return text[:k] + new + text[k + len(old):], True


# ── 1. tokenizer.py ────────────────────────────────────────────────────────
TOK = "src/openpi/models/tokenizer.py"
TOK_PATCH = [
    # subtask 경로
    ('    def tokenize_with_subtask(self, prompt: str, state: np.ndarray, subtask: str):',
     '    def tokenize_with_subtask(self, prompt: str, state: np.ndarray, subtask: str,\n'
     '                              mistake: bool | None = None):',
     None),
    ('        head = self._tokenizer.encode(f"State: {state_str}; Subtask:", add_bos=True)',
     f'        # {MARK}: Subtask 앞에 와야 한다 — 슬롯이 causal 이라 생성 시점에 이미 보여야 한다.\n'
     '        meta = "" if mistake is None else f" Mistake: {\'true\' if mistake else \'false\'};"\n'
     '        head = self._tokenizer.encode(f"State: {state_str};{meta} Subtask:", add_bos=True)',
     None),
    # baseline 경로 (PaligemmaTokenizer 안에서만 — tokenize 는 FAST 쪽에도 있다)
    ('    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:',
     '    def tokenize(self, prompt: str, state: np.ndarray | None = None,\n'
     '                 mistake: bool | None = None) -> tuple[np.ndarray, np.ndarray]:',
     "class PaligemmaTokenizer"),
    ('            full_prompt = f"Task: {cleaned_text}, State: {state_str};\\nAction: "',
     f'            # {MARK}\n'
     '            meta = "" if mistake is None else f", Mistake: {\'true\' if mistake else \'false\'}"\n'
     '            full_prompt = f"Task: {cleaned_text}, State: {state_str}{meta};\\nAction: "',
     "class PaligemmaTokenizer"),
]

# ── 2. transforms.py ───────────────────────────────────────────────────────
TR = "src/openpi/transforms.py"
TR_INJECT_ANCHOR = '''@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):'''
TR_INJECT = f'''@dataclasses.dataclass(frozen=True)
class InjectMistake(DataTransformFn):
    """{MARK}: (episode_index, frame_index) -> mistake 플래그.

    starts 는 {{episode_index: 그 에피소드에서 mistake 가 시작되는 frame_index}} 다.
    여기 없는 에피소드는 전부 False (대본 데모).

    dropout 확률로 키 값을 None 으로 둔다 — 그러면 프롬프트에서 필드가 통째로 빠진다.
    pi0.7 이 메타데이터 필드마다 5% 로 하는 것과 같다. 추론 때 필드가 빠져도
    모델이 망가지지 않게 하는 것이 목적이다.

    추론에는 repack 그룹이 돌지 않으므로 이 transform 도 돌지 않는다.
    """

    starts: dict[int, int] = dataclasses.field(default_factory=dict)
    dropout: float = 0.05

    def __call__(self, data: DataDict) -> DataDict:
        if "episode_index" not in data or "frame_index" not in data:
            return data
        if self.dropout > 0 and np.random.random() < self.dropout:
            return {{**data, "mistake": None}}
        ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        fr = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        s = self.starts.get(ep)
        return {{**data, "mistake": bool(s is not None and fr >= s)}}


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):'''

TR_PATCH = [
    (TR_INJECT_ANCHOR, TR_INJECT, None),
    ('    subtask_slot: bool = False',
     '    subtask_slot: bool = False\n'
     f'    # {MARK}: True 면 프롬프트에 Mistake 필드를 넣는다. 학습·추론이 같아야 한다.\n'
     '    mistake_field: bool = False',
     None),
    ('        subtask = data.pop("subtask", None)',
     f'        # {MARK}: 키가 있으면 그 값(None = 드롭됨), 없으면 추론이므로 False.\n'
     '        if "mistake" in data:\n'
     '            mistake = data.pop("mistake")\n'
     '            if mistake is not None:\n'
     '                mistake = bool(mistake)\n'
     '        else:\n'
     '            mistake = False if self.mistake_field else None\n'
     '        subtask = data.pop("subtask", None)',
     None),
    ('            tokens, token_masks, loss_mask = self.tokenizer.tokenize_with_subtask(prompt, state, subtask)',
     '            tokens, token_masks, loss_mask = self.tokenizer.tokenize_with_subtask(\n'
     '                prompt, state, subtask, mistake=mistake)',
     None),
    ('        tokens, token_masks = self.tokenizer.tokenize(prompt, state)',
     '        tokens, token_masks = self.tokenizer.tokenize(prompt, state, mistake=mistake)',
     None),
]

# ── 3. libero_policy.py (EmbodiChainInputs 안에서만) ────────────────────────
POL = "src/openpi/policies/libero_policy.py"
# 앵커는 `return inputs` 하나다 (EmbodiChainInputs 스코프). 넓게 잡으면
# apply_subtask_patch.py 와 서로 앵커를 깨서 적용 순서에 종속된다.
POL_PATCH = [
    ('''        return inputs''',
     f'''        # {MARK}: 이 dict 는 새로 만들어지므로 명시하지 않으면 플래그가 사라진다.
        if "mistake" in data:
            inputs["mistake"] = data["mistake"]

        return inputs''',
     "class EmbodiChainInputs"),
]

# ── 4. config.py ───────────────────────────────────────────────────────────
CFG = "src/openpi/training/config.py"
CFG_PATCH = [
    ('                        "frame_index": "frame_index",',
     '                        "frame_index": "frame_index",\n'
     f'                        # {MARK}: mistake 는 에피소드마다 시작 프레임이 다르다.\n'
     '                        "episode_index": "episode_index",',
     None),
    ('    subtask_sentences: tuple[str, ...] = ()',
     '    subtask_sentences: tuple[str, ...] = ()\n'
     f'    # {MARK}: (episode_index, mistake 시작 frame_index) 쌍. 없는 에피소드는 False.\n'
     '    mistake_starts: tuple[tuple[int, int], ...] = ()\n'
     '    mistake_dropout: float = 0.05',
     None),
    ('''        if self.subtask_sentences:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectSubtask(ends=self.subtask_ends,
                                                  sentences=self.subtask_sentences)],
            )''',
     '''        if self.subtask_sentences:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectSubtask(ends=self.subtask_ends,
                                                  sentences=self.subtask_sentences)],
            )

        # ''' + MARK + ''': 학습 전용. 추론에는 repack 이 돌지 않아 TokenizePrompt 가 False 로 둔다.
        if self.mistake_starts:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectMistake(starts=dict(self.mistake_starts),
                                                  dropout=self.mistake_dropout)],
            )''',
     None),
    ('                            subtask_slot=bool(float(_os.environ.get("PI05_SUBTASK_W", "0")) > 0),',
     '                            subtask_slot=bool(float(_os.environ.get("PI05_SUBTASK_W", "0")) > 0),\n'
     f'                            # {MARK}: 학습·추론에 같은 값이 들어가야 한다.\n'
     '                            mistake_field=_os.environ.get("PI05_MISTAKE", "0") == "1",',
     None),
]


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    total = 0
    for rel, patches in ((TOK, TOK_PATCH), (TR, TR_PATCH), (POL, POL_PATCH), (CFG, CFG_PATCH)):
        p = root / rel
        s = p.read_text()
        n = 0
        for old, new, scope in patches:
            s, did = scoped_replace(s, old, new, scope=scope, path=rel)
            n += did
        p.write_text(s)
        print(f"  {rel:46} {n}/{len(patches)} 적용")
        total += n
    print(f"MISTAKE-PATCH-DONE  {total} 곳")


if __name__ == "__main__":
    main()
