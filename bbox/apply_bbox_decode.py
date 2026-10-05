#!/usr/bin/env python3
"""추론 디코드를 **두 슬롯**(Pen, Subtask)으로 확장한다.

기존 디코드는 loss_mask 의 첫 True 위치에서 14토큰을 생성했다. bbox 를 켜면
슬롯이 둘이고 사이에 "; Subtask:" 텍스트가 끼므로 그대로는 Pen 슬롯만 채운다.

두 슬롯의 간격은 **상수** 다 (사이 텍스트 토큰 수가 고정). 그래서
  bbox 시작 = argmax(loss_mask)
  subtask 시작 = bbox 시작 + BBOX_SLOT + MID_LEN
로 계산할 수 있다.

bbox 슬롯에는 EOS 가 없다 — 길이가 항상 4 로 고정이다. EOS 처리는 subtask 만.
"""
import pathlib, sys

MARK = "RSC_BBOXDEC"
F = "src/openpi/models/pi0.py"

A_CONST = '_RSC_PAD_ID = 0          # RSC_EOSFIX: sentencepiece pad_id (tokenizer 와 같아야 한다)'
B_CONST = (A_CONST + "\n"
           f"# {MARK}: bbox 슬롯. PI05_BBOX=1 이면 Pen 슬롯이 Subtask 앞에 들어간다.\n"
           '_RSC_BBOX_ON = _os.environ.get("PI05_BBOX", "0") == "1"\n'
           "_RSC_BBOX_SLOT = 4\n"
           "# 두 슬롯 사이 '; Subtask:' 의 토큰 수. tokenizer 와 반드시 같아야 한다.\n"
           "_RSC_BBOX_MID = 4\n"
           'logging.info("RSC_BBOX: bbox 슬롯 = %s", _RSC_BBOX_ON)')

A_LOOP = '''        finished = jnp.zeros(tok.shape[0], dtype=bool)
        for k in range(_RSC_SUBTASK_SLOT):
            tok, finished = one(tok, finished, k)'''
B_LOOP = f'''        # {MARK}: bbox 를 켜면 loss_mask 의 첫 True 는 **Pen 슬롯** 이다.
        #   Pen 4칸을 먼저 채우고, 고정 간격만큼 건너뛴 뒤 Subtask 14칸을 채운다.
        #   Pen 슬롯은 길이가 고정이라 EOS 를 보지 않는다 (always_open=True).
        if _RSC_BBOX_ON:
            done0 = jnp.zeros(tok.shape[0], dtype=bool)
            for k in range(_RSC_BBOX_SLOT):
                tok, done0 = one(tok, done0, k, always_open=True)
            sub0 = _RSC_BBOX_SLOT + _RSC_BBOX_MID
        else:
            sub0 = 0

        finished = jnp.zeros(tok.shape[0], dtype=bool)
        for k in range(_RSC_SUBTASK_SLOT):
            tok, finished = one(tok, finished, sub0 + k, k_eos=k)'''

A_ONE = '''        def one(tok, finished, k):'''
B_ONE = f'''        def one(tok, finished, k, *, always_open=False, k_eos=None):
            """{MARK}: k 는 **슬롯 시작으로부터의 오프셋**. k_eos 는 EOS 강제 판정용
            (subtask 슬롯 안에서 몇 번째인가). always_open 이면 EOS/pad 처리를 끈다."""'''

A_EOS = '''            # RSC_EOSFIX: EOS 뒤는 학습에서 전부 pad 였다. 생성도 그렇게 맞춘다.
            if k == _RSC_SUBTASK_SLOT - 1:
                # 마지막 칸까지 EOS 가 없으면 EOS 를 강제한다 (학습의 truncation 과 같다).
                nxt = jnp.where(finished[:, None], _RSC_PAD_ID, _RSC_EOS_ID)
            else:
                nxt = jnp.where(finished[:, None], _RSC_PAD_ID, nxt)
            finished = finished | (nxt[:, 0] == _RSC_EOS_ID)'''
B_EOS = f'''            # RSC_EOSFIX: EOS 뒤는 학습에서 전부 pad 였다. 생성도 그렇게 맞춘다.
            # {MARK}: bbox 슬롯은 길이 고정이라 이 처리를 건너뛴다.
            if not always_open:
                if k_eos is not None and k_eos == _RSC_SUBTASK_SLOT - 1:
                    # 마지막 칸까지 EOS 가 없으면 EOS 를 강제한다 (학습의 truncation 과 같다).
                    nxt = jnp.where(finished[:, None], _RSC_PAD_ID, _RSC_EOS_ID)
                else:
                    nxt = jnp.where(finished[:, None], _RSC_PAD_ID, nxt)
                finished = finished | (nxt[:, 0] == _RSC_EOS_ID)'''


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    p = root / F
    s = p.read_text()
    if MARK in s:
        print(f"  {F}: 이미 적용됨")
        return
    for a in (A_CONST, A_LOOP, A_ONE, A_EOS):
        n = s.count(a)
        if n != 1:
            raise SystemExit(f"★ {F}: 앵커가 {n} 번 (1 이어야 함)\n{a.splitlines()[0][:90]}")
    s = (s.replace(A_CONST, B_CONST, 1).replace(A_ONE, B_ONE, 1)
          .replace(A_EOS, B_EOS, 1).replace(A_LOOP, B_LOOP, 1))
    p.write_text(s)
    print(f"  {F}: ✓ 적용")


if __name__ == "__main__":
    main()
