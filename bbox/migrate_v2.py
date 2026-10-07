#!/usr/bin/env python3
"""**이미 설치된 소스**를 2026-10-07 리뷰 수정본으로 올린다.

왜 따로 필요한가
  패치 스크립트는 파일에 마커가 있으면 "이미 적용됨" 으로 건너뛴다. 그 판정은
  *무엇이* 적용됐는지는 안 본다 — 그래서 옛 커밋이 심어 둔 코드가 있는 머신에
  새 패치를 돌리면 **전부 건너뛰고 옛 코드가 그대로 남는다**. 깨끗한 설치만
  검증하면 못 잡는다 (2026-10-07 리뷰 P1).

  고쳐야 할 것이 '없던 코드를 넣는' 것이 아니라 '있는 코드를 바꾸는' 것이므로
  rep() 의 앵커 방식으로는 표현할 수 없다. 마커 검사보다 **먼저** 돌린다.

세 가지를 다룬다 (나머지 두 건은 마이그레이션이 필요 없다)
  M1 tokenizer.py  손목 슬롯 변수를 분기 밖에서 초기화  (bbox OFF 크래시)
  M2 pi0.py        bbox 슬롯의 pad 제외 복원            (누락 라벨을 감독)
  M3 pi0.py        집기 가중을 flow 에만                (flow+CE 전체였다)
  —  model.py 의 필드 위치는 설치 결과가 같다 (앵커만 바꿨다).
  —  config.py 의 집기 블록도 설치 결과가 같다.
  —  prepare_mistake_config.py 는 생성기라 설치본이 없다.

마이그레이션 뒤의 파일은 **깨끗한 설치와 바이트 단위로 같아야 한다**.
migrate_v2_test.py 가 그걸 확인한다.
"""

# ── M1. tokenizer.py — 손목 변수 분기 밖 초기화 ───────────────────────────
M1_DONE = "RSC_WBOX: 분기 밖 초기화"
M1_A = "        pad_id0 = self._tokenizer.pad_id() if self._tokenizer.pad_id() >= 0 else 0"
M1_B = (M1_A + "\n"
        "        wbox_body, wmid = [], []        # RSC_WBOX: 분기 밖 초기화")
M1_A2 = "                wbox_body, wmid = [], []\n"
M1_B2 = "                wbox_body, wmid = [], []      # 손목 끔\n"

# ── M2. pi0.py — bbox 슬롯 pad 제외 복원 ──────────────────────────────────
M2_A = """        # RSC_NOBOX: '박스 없음' 도 <loc0000> 네 개로 **가르친다**. pad 를 빼는 처리를
        #   없앴다 — 그게 "그 자리에 무엇을 넣어도 벌점 없음" 을 14.9% 학습시켰고,
        #   추론에서 27% 가 <loc> 아닌 토큰(BOS, 'Sub')을 내는 원인으로 보인다.
        #   학습 라벨에는 pad 가 더 이상 안 들어온다 (visible 3상태)."""
M2_B = """        # RSC_NOBOX: '박스 없음' 은 <loc0000>(id 256000) 네 개로 **가르친다**.
        #   pad(id 0) 제외는 **그대로 둔다** — 둘은 id 가 달라 충돌하지 않는다.
        #   라벨 파일이 없는 프레임은 여전히 빈 슬롯(pad)으로 가는데, 그걸 CE 대상으로
        #   삼으면 '패딩을 맞혀라' 를 가르치게 된다. 누락은 verify 로 따로 잡는다.
        if _RSC_BBOX_ON:
            is_bbox = (jnp.arange(tgt.shape[1]) < n_bbox)[None, :]
            msk = msk * (~(is_bbox & (tgt == _RSC_PAD_ID))).astype(jnp.float32)"""

# ── M3. pi0.py — 집기 가중을 flow 에만 ────────────────────────────────────
M3_A = """        total = flow + _RSC_SUBTASK_W * ce[:, None]
        # RSC_GRASPW: 집기 같은 드문 결정적 구간을 키운다. 없으면 1.0 이다.
        if observation.loss_weight is not None:
            total = total * observation.loss_weight[:, None]"""
M3_B = """        # RSC_GRASPW: 집기 같은 드문 결정적 구간의 **action 회귀만** 키운다.
        #   CE(subtask·bbox)까지 같이 키우면 언어 쪽 균형이 틀어진다 —
        #   우리가 늘리고 싶은 것은 그 순간의 action 정밀도다.
        if observation.loss_weight is not None:
            flow = flow * observation.loss_weight[:, None]
        total = flow + _RSC_SUBTASK_W * ce[:, None]"""

# 파일별 규칙: (이름, 선행조건 마커, 이미 끝났는지 보는 문자열, [(옛, 새), ...])
RULES = {
    "src/openpi/models/tokenizer.py": [
        ("M1 손목 변수 분기 밖 초기화", "RSC_WBOX", M1_DONE,
         [(M1_A, M1_B), (M1_A2, M1_B2)]),
    ],
    "src/openpi/models/pi0.py": [
        ("M2 bbox 슬롯 pad 제외 복원", "RSC_NOBOX", None, [(M2_A, M2_B)]),
        ("M3 집기 가중을 flow 에만",    "RSC_GRASPW", None, [(M3_A, M3_B)]),
    ],
}


def migrate_text(rel, s):
    """(새 텍스트, [설명...]) 을 돌려준다. 해당 없으면 원문 그대로."""
    done = []
    for name, need, already, pairs in RULES.get(rel, []):
        if need not in s:
            continue                      # 그 패치가 아직 안 깔렸다 — 할 일 없다
        if already is not None and already in s:
            continue                      # 이미 새 형태
        old, _ = pairs[0]
        if old not in s:
            continue                      # 이미 새 형태 (또는 손으로 고침)
        for a, b in pairs:
            n = s.count(a)
            if n != 1:
                raise SystemExit(
                    f"★ {rel}: 마이그레이션 '{name}' 의 옛 코드가 {n} 번이다 "
                    f"(1 이어야 함) — 손으로 확인할 것")
            s = s.replace(a, b, 1)
        done.append(name)
    return s, done


def migrate_path(p, rel):
    """실제 파일에 적용한다. 바뀐 마이그레이션 이름들을 돌려준다."""
    s = p.read_text()
    s2, done = migrate_text(rel, s)
    if done:
        p.write_text(s2)
    return done


def run(root, rels):
    """패치 스크립트가 main() 맨 앞에서 부른다."""
    for rel in rels:
        p = root / rel
        if not p.exists():
            continue
        for name in migrate_path(p, rel):
            print(f"  {rel}: ↑ 마이그레이션 — {name}")
