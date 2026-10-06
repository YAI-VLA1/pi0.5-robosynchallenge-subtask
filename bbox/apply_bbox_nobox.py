#!/usr/bin/env python3
"""'박스 없음' 을 **가르친다**. (CE 에서 빼지 않는다)

무엇이 문제였나
  펜이 화면 밖인 프레임(14.9%)의 Pen 슬롯을 pad 로 채우고 CE 에서 뺐다.
  그러면 모델은 "그 자리에 무엇을 넣어도 벌점이 없다" 를 15% 학습한다.
  실제로 추론에서 27% 가 <loc> 아닌 토큰(BOS 526회, 'Sub' 300회)을 냈다.

어떻게 고치나
  라벨의 visible 열이 3상태가 됐다.
      0  추론용 빈 슬롯 (라벨 파일에는 없다 — 모델이 채운다)
      1  실제 박스
      2  "박스 없음" -> <loc0000> 네 개로 가르친다
  네 칸 모두 0 은 ymax<=ymin 이라 실제 박스로는 절대 안 나오므로 안전한 표식이다.
  CE 의 pad 제외도 없앤다 — 이제 모든 학습 프레임이 감독된다.
"""
import pathlib, sys

MARK = "RSC_NOBOX"

TOK = "src/openpi/models/tokenizer.py"
TOK_A = '''            b = np.asarray(bbox).reshape(-1).astype(int)
            if b.size >= 5 and b[4] > 0:
                # <locNNNN> 는 사전에 이미 있다. id = 256000 + N 로 바로 쓴다.
                base = self._tokenizer.piece_to_id("<loc0000>")
                bbox_body = [base + int(min(max(v, 0), 1023)) for v in b[:4]]
            else:
                bbox_body = [pad_id0] * self.BBOX_SLOT'''
TOK_B = f'''            b = np.asarray(bbox).reshape(-1).astype(int)
            # {MARK}: visible 3상태. 1=실제 박스, 2="박스 없음"(가르친다), 0=추론 빈 슬롯.
            #   1 과 2 는 둘 다 <loc> 네 개를 내보낸다 — 2 는 전부 <loc0000> 이다.
            #   pad 로 두고 CE 에서 빼면 "아무거나 넣어도 벌점 없음" 을 가르치게 된다.
            if b.size >= 5 and b[4] > 0:
                base = self._tokenizer.piece_to_id("<loc0000>")
                bbox_body = [base + int(min(max(v, 0), 1023)) for v in b[:4]]
            else:
                bbox_body = [pad_id0] * self.BBOX_SLOT      # 추론: 빈 슬롯'''

PI0 = "src/openpi/models/pi0.py"
PI0_A = '''        # RSC_BBOXFIX: 박스가 안 보이는 프레임은 슬롯이 pad 다 — 감독에서만 뺀다
        #   (loss mask 는 구조용이라 그대로 둔다).
        if _RSC_BBOX_ON:
            is_bbox = (jnp.arange(tgt.shape[1]) < n_bbox)[None, :]
            msk = msk * (~(is_bbox & (tgt == _RSC_PAD_ID))).astype(jnp.float32)'''
PI0_B = f'''        # {MARK}: '박스 없음' 도 <loc0000> 네 개로 **가르친다**. pad 를 빼는 처리를
        #   없앴다 — 그게 "그 자리에 무엇을 넣어도 벌점 없음" 을 14.9% 학습시켰고,
        #   추론에서 27% 가 <loc> 아닌 토큰(BOS, 'Sub')을 내는 원인으로 보인다.
        #   학습 라벨에는 pad 가 더 이상 안 들어온다 (visible 3상태).'''


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    for rel, a, b in ((TOK, TOK_A, TOK_B), (PI0, PI0_A, PI0_B)):
        p = root / rel
        s = p.read_text()
        if MARK in s:
            print(f"  {rel}: 이미 적용됨"); continue
        n = s.count(a)
        if n != 1:
            raise SystemExit(f"★ {rel}: 앵커가 {n} 번 (1 이어야 함)\n{a.splitlines()[0][:80]}")
        p.write_text(s.replace(a, b, 1))
        print(f"  {rel}: ✓ 적용")
    print("NOBOX-PATCH-DONE")


if __name__ == "__main__":
    main()
