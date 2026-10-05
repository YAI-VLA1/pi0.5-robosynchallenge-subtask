#!/usr/bin/env python3
"""bbox 마이그레이션의 네 상태를 검사한다. (외부 리뷰)

  fresh / 옛 단일 블록 / 이미 최종형 / 중복(직전 업데이트 버그)
각각에서 **최종적으로 pop 이 정확히 1번** 이어야 한다.
"""
import pathlib, sys
sys.path.insert(0, "/root/rsc_recover")
import bbox_migrate
import importlib.util as u
sp = u.spec_from_file_location("ab", "/root/rsc_recover/apply_bbox.py")
ab = u.module_from_spec(sp); sp.loader.exec_module(ab)

HEAD = '''@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    mistake_field: bool = False
    bbox_slot: bool = False

    def __call__(self, data):
'''
TAIL = '''        subtask = data.pop("subtask", None)
        return data


class Other:
    pass
'''
OLD = '''        # RSC_BBOX: 라벨이 없으면(추론) 빈 슬롯을 만들어 모델이 채우게 한다.
        bbox = data.pop("bbox", None)
        if bbox is None and self.bbox_slot:
            bbox = np.zeros(5, np.int32)
'''
FINAL = '''        # RSC_BBOX: 라벨이 없으면(추론) 빈 슬롯을 만들어 모델이 채우게 한다.
        bbox = data.pop("bbox", None)
        # RSC_BBOX: bbox 를 끄면 라벨이 들어와도 버린다 — 안 그러면 토크나이저는
        #   슬롯을 만들고 모델은 없다고 가정해 CE 인덱스가 어긋난다.
        if not self.bbox_slot:
            bbox = None
        elif bbox is None:
            bbox = np.zeros(5, np.int32)
'''

CASES = {
    "① fresh (bbox 코드 없음)": HEAD + TAIL,
    "② 옛 단일 블록":            HEAD + OLD + TAIL,
    "③ 이미 최종형":             HEAD + FINAL + TAIL,
    "④ 중복 (직전 버그)":        HEAD + OLD + FINAL + TAIL,
}

# 삽입 패치 하나만 꺼내 쓴다 (목록에서 subtask 앵커를 쓰는 것)
INS = next(x for x in ab.TR_P if x[0] == '        subtask = data.pop("subtask", None)')

ok = {}
for tag, src in CASES.items():
    try:
        s, why = bbox_migrate.migrate(src)
        s, did = ab.rep(s, *INS[:2], scope=INS[2], path="테스트")
        bbox_migrate.check_single_pop(s, "테스트")
        n = s.count('bbox = data.pop("bbox"')
        has_final = "if not self.bbox_slot:" in s
        ok[tag] = (n == 1 and has_final)
        print(f"{tag:26} {why:38} pop {n} · 최종형 {'있음' if has_final else '없음'}")
    except SystemExit as e:
        ok[tag] = False
        print(f"{tag:26} ★ {e}")

print()
for k, v in ok.items():
    print(f"  {'OK ' if v else '✗  '} {k}")
print("\nBBOX-MIGRATE-TEST " + ("OK" if all(ok.values()) else "FAIL"))
if not all(ok.values()):
    raise SystemExit(1)
