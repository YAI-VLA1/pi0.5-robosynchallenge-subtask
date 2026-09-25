#!/usr/bin/env python3
"""phase -> GT subtask 문장.

왜 자연어로 쓰나
  구간 이름(`left_pre1_to_takeover`)을 그대로 토큰화하면 PaliGemma 의 언어 prior 가
  거의 안 붙는다. 자연어로 쓰면 (a) 사전학습 의미를 쓰고 (b) 나중에 10개 태스크를
  합칠 때 "close the left gripper" 같은 구절이 태스크를 넘어 공유된다.

작성 규칙
  · 팔을 왼/오른쪽으로 명시한다 (delta action 이 팔별로 갈리므로 중요하다)
  · 동사 + 목적어 + (필요하면) 목표 위치. 8~12 토큰을 넘기지 않는다
  · 같은 종류의 동작은 같은 표현을 쓴다 (태스크 간 공유를 위해)
  · 두 구간이 동시에 도는 phase 는 둘 다 담되 주된 쪽을 앞에 둔다

한계
  문장은 내가 썼다. 데이터에 있던 것이 아니다. 구간 이름과 그 구간에서 실제로
  일어나는 일(2026-09-25 물체 궤적 검증)에 근거했지만 표현 자체는 임의다.
"""
from __future__ import annotations

# items_handover — phase id -> 문장. phases.py 가 만드는 15개와 순서가 같아야 한다.
ITEMS_HANDOVER = {
    0:  "open both grippers",
    1:  "move the right arm toward the pen",
    2:  "lower the right gripper onto the pen",
    3:  "close the right gripper on the pen",
    4:  "lift the pen with the right arm",
    5:  "move the right arm to the handover pose",
    6:  "move the left arm toward the pen",
    7:  "bring the left gripper around the pen",
    8:  "close the left gripper on the pen",
    9:  "open the right gripper to release the pen",
    10: "move the right arm back",
    11: "carry the pen with the left arm",
    12: "move the pen over the holder",
    13: "open the left gripper to drop the pen in the holder",
    14: "lift the left arm away from the holder",
}

TABLES = {"items_handover": ITEMS_HANDOVER}


def sentences(task: str, n_phases: int) -> list[str]:
    t = TABLES.get(task)
    if t is None:
        raise KeyError(f"{task} 의 subtask 문장이 아직 없다. sentences.py 에 추가할 것")
    if sorted(t) != list(range(n_phases)):
        raise ValueError(f"{task}: 문장 {len(t)}개인데 phase 는 {n_phases}개다")
    return [t[i] for i in range(n_phases)]


if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from phases import phases

    task = sys.argv[2] if len(sys.argv) > 2 else "items_handover"
    ph = phases(pathlib.Path(sys.argv[1]), task)
    for p, s in zip(ph, sentences(task, len(ph))):
        print(f"{p['id']:>2}  {p['start']:>4}-{p['end']:<4}  {p['name'][:40]:40s}  \"{s}\"")
