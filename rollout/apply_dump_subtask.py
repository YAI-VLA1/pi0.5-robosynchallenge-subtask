#!/usr/bin/env python3
"""롤아웃 중 생성되는 subtask 문장을 파일로 남긴다 (RSC_DUMP_SUBTASK). 멱등, --check.

왜
  `_rsc_generate_subtask` 는 호출마다 슬롯을 새로 채우는데, 그 결과가 아무데도
  안 남는다. 평가에서 모델이 실제로 무슨 단계라고 말하는지 봐야 실패를 읽을 수 있다.
  성공/실패 판정만으로는 "왼손 그리퍼를 닫아야 할 때 닫으라고 했는가"를 알 수 없다.

어떻게
  `sample_actions` 는 jit 안에서 돈다. 그래서 print 가 아니라 `io_callback` 으로
  호스트에 토큰 id 를 넘겨 파일에 append 한다. 환경변수 `PI05_DUMP_SUBTASK` 에
  경로를 주면 켜지고, 없으면 코드 경로 자체가 실행되지 않는다(비용 0).

  디코드는 오프라인에서 한다 — 콜백 안에서 sentencepiece 를 부르면 느리고,
  롤아웃 루프를 건드리고 싶지 않다.

사용:  python3 apply_dump_subtask.py <repo 루트> [--check]
       평가 시  PI05_DUMP_SUBTASK=/workspace/subtask_gen.jsonl
"""
from __future__ import annotations

import pathlib
import sys

MARK = "RSC_DUMP_SUBTASK"
PI0 = "policy/pi05/src/openpi/models/pi0.py"

HDR_A = "_RSC_EOS_ID = 1          # PaliGemma sentencepiece eos_id"
HDR_B = '''_RSC_EOS_ID = 1          # PaliGemma sentencepiece eos_id

# {mark}: 롤아웃 중 생성된 subtask 토큰을 파일로 남긴다. 경로가 없으면 완전히 꺼진다.
_RSC_DUMP_PATH = _os.environ.get("PI05_DUMP_SUBTASK", "")
# RSC_BBOXDUMP: 이 블록은 bbox 패치보다 **앞 줄**에 들어가므로 _RSC_BBOX_ON 을
#   아직 못 쓴다. 환경변수를 직접 읽는다 (값은 같다).
_RSC_DUMP_LEN = (8 if _os.environ.get("PI05_BBOX", "0") == "1" else 0) + _RSC_SUBTASK_SLOT


def _rsc_dump_tokens(ids):
    """io_callback 대상. jit 안에서 호출되므로 여기서는 파일 append 만 한다."""
    import numpy as _np
    try:
        with open(_RSC_DUMP_PATH, "a") as f:
            for row in _np.asarray(ids).reshape(-1, _RSC_DUMP_LEN):
                f.write(",".join(str(int(x)) for x in row) + "\\n")
    except Exception:       # noqa: BLE001  로깅이 롤아웃을 죽이면 안 된다
        pass'''.format(mark=MARK)

# 앵커를 `return dataclasses.replace(...)` 한 줄로 좁혔다. 예전에는 디코드
# 루프 본문까지 묶었는데, EOS 수정(RSC_EOSFIX)이 루프를 바꾸면서 앵커가 깨졌다.
# 덤프는 '생성이 끝난 tok' 만 보면 되므로 루프 모양에 의존할 이유가 없다.
GEN_A = '''        return dataclasses.replace(obs, tokenized_prompt=tok)'''
GEN_B = '''        # {mark}: 생성된 슬롯을 그대로 흘려보낸다 (디코드는 오프라인).
        if _RSC_DUMP_PATH:
            # RSC_BBOXDUMP: bbox 를 켜면 loss mask 의 첫 True 는 Pen 슬롯이다.
            #   Pen 4 | "; Subtask:" 4 | Subtask 14 를 **통째로** 뜬다.
            #   예전처럼 14칸만 뜨면 bbox 4 + 사이 4 + subtask 앞 6 이 찍혀
            #   문장이 잘린 채로 나온다.
            slot = jnp.take_along_axis(tok, start[:, None] + jnp.arange(_RSC_DUMP_LEN), axis=1)
            jax.experimental.io_callback(_rsc_dump_tokens, None, slot)

        return dataclasses.replace(obs, tokenized_prompt=tok)'''.format(mark=MARK)

IMP_A = "import jax.numpy as jnp"
IMP_B = "import jax.experimental\nimport jax.numpy as jnp"

EDITS = [(HDR_A, HDR_B), (IMP_A, IMP_B), (GEN_A, GEN_B)]


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    check = "--check" in sys.argv
    p = root / PI0
    if not p.exists():
        sys.exit(f"{PI0}: 파일 없음")
    s = p.read_text()
    if MARK in s:
        print(f"  {PI0}: 이미 적용됨")
        return
    for a, _ in EDITS:
        n = s.count(a)
        if n != 1:
            sys.exit(f"  {PI0}: ✗ 앵커가 {n}곳 — {a.splitlines()[0][:70]!r}")
    if check:
        print(f"  {PI0}: 미적용 (적용 가능)")
        sys.exit(1)
    for a, b in EDITS:
        s = s.replace(a, b, 1)
    p.write_text(s)
    print(f"  {PI0}: ✓ 적용")
    print("\n평가 시 PI05_DUMP_SUBTASK=<경로> 를 주면 토큰 id 가 한 줄씩 쌓인다.")
    print("디코드: python3 decode_dump.py <경로>")


if __name__ == "__main__":
    main()
