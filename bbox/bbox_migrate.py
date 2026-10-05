#!/usr/bin/env python3
"""TokenizePrompt 안의 bbox 블록을 **하나로 정리**한다.

패치 목록에 넣으면 안 된다 — rep() 는 앵커가 정확히 1번 있어야 하므로
옛 블록이 없는 정상 상태(fresh / 이미 최종형)까지 오류가 된다. 조건부로 돈다.

다뤄야 하는 상태 네 가지
  fresh        bbox 코드 없음              -> 건너뜀
  옛 단일 블록   `if bbox is None and ...`   -> 지움 (삽입 패치가 최종형을 넣는다)
  이미 최종형    `if not self.bbox_slot:`    -> 건너뜀
  중복          pop 이 두 번                -> 전부 지우고 삽입 패치가 하나만 넣는다
"""
import re

POP = 'bbox = data.pop("bbox"'

# 옛 블록 (2026-10-05 이전 형태)
OLD_BLOCK = (
    '        # RSC_BBOX: 라벨이 없으면(추론) 빈 슬롯을 만들어 모델이 채우게 한다.\n'
    '        bbox = data.pop("bbox", None)\n'
    '        if bbox is None and self.bbox_slot:\n'
    '            bbox = np.zeros(5, np.int32)\n'
)

# 최종형. 주석 문구가 MARK 에 따라 달라질 수 있어 정규식으로 찾는다.
FINAL_RE = re.compile(
    r"[ \t]*#[^\n]*\n(?:[ \t]*#[^\n]*\n)*"                 # 앞 주석 줄들
    r'[ \t]*bbox = data\.pop\("bbox", None\)\n'
    r"(?:[ \t]*#[^\n]*\n)*"                                # 사이 주석
    r"[ \t]*if not self\.bbox_slot:\n[ \t]*bbox = None\n"
    r"[ \t]*elif bbox is None:\n[ \t]*bbox = np\.zeros\(5, np\.int32\)\n"
)


def _body(s):
    i = s.find("class TokenizePrompt")
    if i < 0:
        return None, None, None
    j = s.find("\nclass ", i + 1)
    j = len(s) if j < 0 else j
    return i, j, s[i:j]


def migrate(s):
    """(새 텍스트, 설명) 을 돌려준다."""
    i, j, body = _body(s)
    if body is None:
        return s, "TokenizePrompt 없음"
    npop = body.count(POP)
    if npop == 0:
        return s, "fresh (bbox 코드 없음)"
    nold = body.count(OLD_BLOCK)
    if npop == 1 and nold == 0:
        return s, "이미 최종형 — 건너뜀"

    nb = FINAL_RE.sub("", body.replace(OLD_BLOCK, ""))
    left = nb.count(POP)
    if left:
        raise SystemExit(f"★ bbox 블록을 다 못 지웠다 (남은 pop {left}) — 손으로 확인할 것")
    return s[:i] + nb + s[j:], f"정리 (pop {npop} -> 0, 삽입 패치가 최종형 하나를 넣는다)"


def check_single_pop(s, path=""):
    _, _, body = _body(s)
    n = 0 if body is None else body.count(POP)
    if n != 1:
        raise SystemExit(f"★ {path}: TokenizePrompt 안의 bbox pop 이 {n} 번이다 (1 이어야 함)")
