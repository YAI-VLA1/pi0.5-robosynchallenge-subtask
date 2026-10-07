#!/usr/bin/env python3
"""토크나이저가 **네 조합 전부**에서 도는지 본다.

  bbox=None(대조군) / bbox 만 / bbox+wrist / 박스없음 둘 다

왜 필요한가
  손목 슬롯을 넣으면서 wmid·wbox_body 를 `bbox is not None` 분기 안에서만
  초기화했더니, 대조군(bbox=None)에서 **UnboundLocalError** 가 났다.
  슬롯 조합마다 토큰 조립 경로가 다르므로 전부 돌려 봐야 한다.

실제 소스에서 함수 본문을 떼어 더미 tokenizer 로 돌린다 — 모델도 sentencepiece 도 없이.
"""
import pathlib, re, sys
import numpy as np

PI05 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
src = (PI05 / "src/openpi/models/tokenizer.py").read_text()
m = re.search(r"    def tokenize_with_subtask\(.*?\n(?=    def )", src, re.S)
if not m:
    raise SystemExit("★ tokenize_with_subtask 를 못 찾았다")

STUB = '''
import numpy as np, logging
class _T:
    def encode(self, s, add_bos=False): return [1]*max(1,len(s)//4)
    def eos_id(self): return 1
    def pad_id(self): return 0
    def piece_to_id(self, s): return 256000
class Tk:
    SUBTASK_SLOT=14; BBOX_SLOT=4; BBOX_MID="; Subtask:"; WBOX_MID=" Wrist:"
    _max_len=400
    def __init__(self): self._tokenizer=_T()
''' + m.group(0)

ns = {}
exec(STUB, ns)
t = ns["Tk"]()
cases = [
    ("bbox=None (대조군)", dict(bbox=None),                       14),
    ("bbox 만",            dict(bbox=np.array([1,2,3,4,1])),      18),
    ("bbox + wrist",       dict(bbox=np.array([1,2,3,4,1]),
                                wbox=np.array([5,6,7,8,1])),      22),
    ("박스없음 둘 다",      dict(bbox=np.array([0,0,0,0,2]),
                                wbox=np.array([0,0,0,0,2])),      22),
]
ok = {}
for tag, kw, want in cases:
    try:
        tok, msk, lm = t.tokenize_with_subtask("x", np.zeros(14), "hi",
                                               mistake=False, **kw)
        n = int(lm.sum())
        ok[tag] = (n == want)
        print(f"  {tag:20} 유효토큰 {int(msk.sum()):3} · loss mask {n:2} (기대 {want})")
    except Exception as e:
        ok[tag] = False
        print(f"  {tag:20} ★ {type(e).__name__}: {e}")
print()
for k, v in ok.items():
    print(f"  {'OK ' if v else '✗  '} {k}")
print("\nTOKENIZER-SLOTS " + ("OK" if all(ok.values()) else "FAIL"))
if not all(ok.values()):
    raise SystemExit(1)
