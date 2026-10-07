#!/usr/bin/env python3
"""migrate_v2 라운드트립 — 새 코드를 옛 코드로 되돌린 뒤 올리면 원래대로인가.

  bash setup/apply_patches.sh <repo> --all      # 깨끗한 설치 = 정답
  python3 bbox/migrate_v2_test.py <repo>/policy/pi05

설치본을 '정답' 으로 두고 각 마이그레이션을 **거꾸로** 적용해 옛 상태를 만든 뒤
migrate_v2 로 올려서 정답과 바이트 단위로 같은지 본다. 셋을 따로, 그리고 한꺼번에.
'옛 상태' 를 손으로 흉내 내지 않고 실제 치환 쌍을 뒤집어 쓰므로 패치가 바뀌면
이 테스트도 같이 따라간다.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import migrate_v2 as M

TOK = "src/openpi/models/tokenizer.py"
PI0 = "src/openpi/models/pi0.py"

# (이름, 파일, 되돌리기 쌍들) — migrate_v2 의 치환을 그대로 뒤집는다
CASES = [
    ("M1 손목 분기 밖 초기화", TOK, [(M.M1_B, M.M1_A), (M.M1_B2, M.M1_A2)]),
    ("M2 pad 제외",            PI0, [(M.M2_B, M.M2_A)]),
    ("M3 flow-only 가중",      PI0, [(M.M3_B, M.M3_A)]),
]


def revert(texts, cases):
    out = dict(texts)
    for name, rel, pairs in cases:
        s = out[rel]
        for a, b in pairs:
            if s.count(a) != 1:
                raise SystemExit(f"★ {name}: 설치본에서 새 코드를 {s.count(a)} 번 찾았다 "
                                 f"(1 이어야 함) — --all 설치가 끝난 트리를 줄 것")
            s = s.replace(a, b, 1)
        out[rel] = s
    return out


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    gold = {rel: (root / rel).read_text() for rel in (TOK, PI0)}
    for rel, s in gold.items():
        if "RSC_WBOX" not in s and "RSC_GRASPW" not in s and "RSC_NOBOX" not in s:
            raise SystemExit(f"★ {rel} 에 bbox 패치가 안 깔려 있다 — --all 로 설치할 것")

    runs = [([c], c[0]) for c in CASES] + [(CASES, "셋 한꺼번에")]
    bad = 0
    for cases, label in runs:
        old = revert(gold, cases)
        got, names = {}, []
        for rel in (TOK, PI0):
            s, done = M.migrate_text(rel, old[rel])
            got[rel], _ = s, names.extend(done)
        ok = all(got[rel] == gold[rel] for rel in (TOK, PI0))
        # 되돌린 쪽은 반드시 달랐어야 한다 — 아니면 테스트가 아무것도 안 본 것이다
        moved = any(old[rel] != gold[rel] for rel in (TOK, PI0))
        if not moved:
            ok = False; names = ["되돌리기가 아무것도 안 바꿨다"]
        print(f"{'OK ' if ok else 'FAIL'} {label:24} 적용 {len(names)}건: {', '.join(names) or '없음'}")
        bad += not ok

    # 멱등: 이미 새 형태에 또 돌려도 변화가 없어야 한다
    noop = all(M.migrate_text(rel, gold[rel]) == (gold[rel], []) for rel in (TOK, PI0))
    print(f"{'OK ' if noop else 'FAIL'} 멱등 (새 형태에 재실행)")
    bad += not noop

    print("MIGRATE-V2 OK" if not bad else f"MIGRATE-V2 FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
