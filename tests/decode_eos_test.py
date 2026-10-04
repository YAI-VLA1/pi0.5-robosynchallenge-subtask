#!/usr/bin/env python3
"""EOS 처리 검증. (리뷰 R2)

실제 pi0.py 의 디코드 루프 본문을 그대로 베껴 와서, 토큰을 내는 부분만
가짜 LM 으로 바꾼다. 모델 전체를 띄우지 않고 '어떤 토큰이 슬롯에 남는가'만 본다.

보는 것
  ① 첫 칸에서 EOS -> 나머지 전부 pad
  ② 중간에서 EOS  -> 그 뒤 전부 pad
  ③ 끝까지 EOS 없음 -> 마지막 칸이 EOS 로 강제됨
  ④ 배치에서 한 샘플만 먼저 끝나도 서로 간섭하지 않음
"""
import jax.numpy as jnp
import numpy as np

SLOT, EOS, PAD = 14, 1, 0


def decode(fake_tokens):
    """fake_tokens[b][k] = k 번째 칸에서 LM 이 내는 토큰. pi0.py 와 같은 로직."""
    b = len(fake_tokens)
    tok = jnp.zeros((b, SLOT), dtype=jnp.int32)
    start = jnp.zeros(b, dtype=jnp.int32)
    finished = jnp.zeros(b, dtype=bool)
    for k in range(SLOT):
        nxt = jnp.array([[fake_tokens[i][k]] for i in range(b)], dtype=jnp.int32)
        if k == SLOT - 1:
            nxt = jnp.where(finished[:, None], PAD, EOS)
        else:
            nxt = jnp.where(finished[:, None], PAD, nxt)
        finished = finished | (nxt[:, 0] == EOS)
        tok = jnp.put_along_axis(tok, (start + k)[:, None], nxt, axis=1, inplace=False)
    return np.asarray(tok)


def main():
    cases = {
        "① 첫 칸 EOS":      [[EOS] + [7] * (SLOT - 1)],
        "② 중간(4) EOS":    [[5, 6, 7, EOS] + [9] * (SLOT - 4)],
        "③ EOS 없음":       [[3] * SLOT],
    }
    ok = {}
    for name, ft in cases.items():
        out = decode(ft)[0]
        print(f"{name:16} {out.tolist()}")
        if name.startswith("①"):
            ok[name] = out[0] == EOS and (out[1:] == PAD).all()
        elif name.startswith("②"):
            ok[name] = out[3] == EOS and (out[4:] == PAD).all() and (out[:3] == [5, 6, 7]).all()
        else:
            ok[name] = out[-1] == EOS and (out[:-1] == 3).all()

    # ④ 섞인 배치
    out = decode([[EOS] + [7] * (SLOT - 1), [5, 6, EOS] + [9] * (SLOT - 3), [3] * SLOT])
    print(f"{'④ 혼합 배치':16}")
    for i, row in enumerate(out):
        print(f"   샘플{i} {row.tolist()}")
    ok["④ 혼합 배치"] = (
        out[0][0] == EOS and (out[0][1:] == PAD).all()
        and out[1][2] == EOS and (out[1][3:] == PAD).all() and (out[1][:2] == [5, 6]).all()
        and out[2][-1] == EOS and (out[2][:-1] == 3).all())

    print()
    for k, v in ok.items():
        print(f"  {'OK ' if v else '✗  '} {k}")
    if not all(ok.values()):
        raise SystemExit("DECODE-EOS-TEST FAIL")
    print("\nDECODE-EOS-TEST OK")


if __name__ == "__main__":
    main()
