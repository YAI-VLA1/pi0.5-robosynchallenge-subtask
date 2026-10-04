#!/usr/bin/env python3
"""추론 디코드가 EOS 이후를 pad 로 채우게 한다. (리뷰 R2)

무엇이 문제인가
  학습은 `문장 + EOS + pad...` 로 슬롯을 채우고 CE 는 EOS 까지만 건다.
  추론은 EOS 를 추적하지 않고 14칸을 전부 greedy 로 채운다. 그래서
  **학습에서 본 적 없는 EOS 뒤 자리**에 임의 토큰이 들어가고, 그 토큰을
  flow-matching action head 가 그대로 읽는다.

  슬롯 전체가 attention 에서 유효 토큰이라, 디코드 출력에서 글자만 숨기는
  것으로는 해결되지 않는다 — 토큰 자체를 pad 로 바꿔야 한다.

어떻게 고치나
  샘플별 finished 플래그를 들고 다닌다.
    · 이미 끝난 샘플     -> pad_id
    · 마지막 칸인데 안 끝남 -> EOS 강제 (학습의 truncation 처리와 같다)
  고정 14-step 루프는 그대로 둔다 (jit 모양이 바뀌면 안 된다).
"""
import pathlib, sys

MARK = "RSC_EOSFIX"
F = "src/openpi/models/pi0.py"

A = '''        def one(tok, k):
            text = self.PaliGemma.llm(tok, method="embed")
            full = jnp.concatenate([img_part, text], axis=1)
            (out, _), _ = self.PaliGemma.llm([full, None], mask=attn, positions=positions)
            # 슬롯 k 번째를 맞히는 자리는 (n_img + start + k - 1)
            at_idx = (n_img + start + k - 1)[:, None, None]
            h = jnp.take_along_axis(out, at_idx, axis=1)           # (b, 1, d)
            nxt = jnp.argmax(self.PaliGemma.llm(h, method="decode"), axis=-1)  # (b, 1)
            return jnp.put_along_axis(tok, (start + k)[:, None], nxt, axis=1, inplace=False)

        for k in range(_RSC_SUBTASK_SLOT):
            tok = one(tok, k)'''

B = f'''        def one(tok, finished, k):
            text = self.PaliGemma.llm(tok, method="embed")
            full = jnp.concatenate([img_part, text], axis=1)
            (out, _), _ = self.PaliGemma.llm([full, None], mask=attn, positions=positions)
            # 슬롯 k 번째를 맞히는 자리는 (n_img + start + k - 1)
            at_idx = (n_img + start + k - 1)[:, None, None]
            h = jnp.take_along_axis(out, at_idx, axis=1)           # (b, 1, d)
            nxt = jnp.argmax(self.PaliGemma.llm(h, method="decode"), axis=-1)  # (b, 1)
            # {MARK}: EOS 뒤는 학습에서 전부 pad 였다. 생성도 그렇게 맞춘다.
            if k == _RSC_SUBTASK_SLOT - 1:
                # 마지막 칸까지 EOS 가 없으면 EOS 를 강제한다 (학습의 truncation 과 같다).
                nxt = jnp.where(finished[:, None], _RSC_PAD_ID, _RSC_EOS_ID)
            else:
                nxt = jnp.where(finished[:, None], _RSC_PAD_ID, nxt)
            finished = finished | (nxt[:, 0] == _RSC_EOS_ID)
            tok = jnp.put_along_axis(tok, (start + k)[:, None], nxt, axis=1, inplace=False)
            return tok, finished

        finished = jnp.zeros(tok.shape[0], dtype=bool)
        for k in range(_RSC_SUBTASK_SLOT):
            tok, finished = one(tok, finished, k)'''

A_ID = "_RSC_EOS_ID = 1          # PaliGemma sentencepiece eos_id"
B_ID = ("_RSC_EOS_ID = 1          # PaliGemma sentencepiece eos_id\n"
        f"_RSC_PAD_ID = 0          # {MARK}: sentencepiece pad_id (tokenizer 와 같아야 한다)")


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    p = root / F
    s = p.read_text()
    if MARK in s:
        print(f"  {F}: 이미 적용됨")
        return
    for a in (A, A_ID):
        n = s.count(a)
        if n != 1:
            raise SystemExit(f"★ {F}: 앵커가 {n} 번 (1 이어야 함)\n{a.splitlines()[0][:80]}")
    p.write_text(s.replace(A_ID, B_ID, 1).replace(A, B, 1))
    print(f"  {F}: ✓ 적용")


if __name__ == "__main__":
    main()
