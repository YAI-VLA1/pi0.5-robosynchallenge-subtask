#!/usr/bin/env python3
"""CE 가 **실제로 어느 토큰을 보는지** 검증한다. (외부 리뷰 ②③)

loss mask 에 True 가 18개 있는지 세는 것으로는 부족하다 — 리뷰 지적대로
CE 가 `start + arange(14)` 만 읽으면 mask 가 18개라도 뒤 8칸이 빠진다.
그래서 **gather 인덱스를 직접 재현해** 어떤 토큰이 들어가는지 디코드해 본다.

보는 것
  ① 감독 대상이 bbox 4 + subtask 14 = 18 자리인가 (사이 텍스트 제외)
  ② subtask 의 **EOS 가 포함**되는가 (뒤 8칸이 빠지면 EOS 가 사라진다)
  ③ bbox 가 안 보여도 **슬롯 위치가 안 밀리는가** (loss mask 첫 True = bbox 시작)
  ④ 안 보이는 bbox 는 감독에서 빠지는가 (구조는 유지)
"""
import os, sys
import numpy as np

sys.path.insert(0, "/root/rsc_recover")
BBOX_SLOT, BBOX_MID, SUBTASK_SLOT = 4, 4, 14
EOS, PAD = 1, 0


def gather_idx(start, bbox_on):
    """pi0.py 의 _rsc_subtask_ce 와 **같은** 인덱스를 만든다."""
    if bbox_on:
        offs = np.concatenate([np.arange(BBOX_SLOT),
                               BBOX_SLOT + BBOX_MID + np.arange(SUBTASK_SLOT)])
    else:
        offs = np.arange(SUBTASK_SLOT)
    return start + offs, (BBOX_SLOT if bbox_on else 0)


def supervised(tok, lm, bbox_on):
    start = int(np.argmax(lm.astype(int)))
    idx, n_bbox = gather_idx(start, bbox_on)
    idx = np.clip(idx, 0, len(tok) - 1)
    tgt = tok[idx]
    msk = lm[idx].astype(float)
    in_sub = (np.arange(len(tgt)) >= n_bbox).astype(int)
    eos = ((tgt == EOS) & (in_sub > 0)).astype(int)
    after = np.cumsum(eos) - eos
    msk = msk * ((after == 0) | (in_sub == 0)).astype(float)
    if bbox_on:
        is_b = np.arange(len(tgt)) < n_bbox
        msk = msk * (~(is_b & (tgt == PAD))).astype(float)
    return start, idx, tgt, msk


def main():
    import openpi.models.tokenizer as T
    tk = T.PaligemmaTokenizer(200)
    sp = tk._tokenizer
    st = np.zeros(14, np.float32)
    ok = {}

    LONG0 = "open the left gripper to drop the pen in the holder"
    for tag, bb in (("보임", np.array([296, 916, 492, 1023, 1])),
                    ("안 보임", np.zeros(5, int))):
        tok, m, lm = tk.tokenize_with_subtask("x", st, LONG0,
                                              mistake=False, bbox=bb)
        start, idx, tgt, msk = supervised(tok, lm, True)
        sup = [int(t) for t, w in zip(tgt, msk) if w > 0]
        nloc = sum(1 for t in sup if 256000 <= t <= 257023)
        has_eos = EOS in sup
        print(f"[{tag}] loss mask 첫 True = {start}  ·  감독 토큰 {len(sup)}개 (<loc> {nloc}, EOS {'있음' if has_eos else '없음'})")
        print(f"       {sp.decode([t for t in sup if t != EOS])!r}")
        if tag == "보임":
            ok["① 감독 = bbox4 + subtask(EOS까지)"] = (nloc == 4 and has_eos)
            ok["② 사이 텍스트는 제외"] = all(
                t in (EOS,) or 256000 <= t <= 257023 or
                t in sp.encode(" " + LONG0) for t in sup)
            start_vis = start
        else:
            ok["③ 안 보여도 슬롯 위치 그대로"] = (start == start_vis)
            ok["④ 안 보이는 bbox 는 감독 제외"] = (nloc == 0 and has_eos)

    # ⑤ 예전 버그 재현. **긴 문장**이어야 드러난다 — 짧은 문장은 subtask 앞 6칸
    #    안에 EOS 가 들어와 예전 방식도 우연히 통과한다.
    LONG = "open the left gripper to drop the pen in the holder"
    tok, m, lm = tk.tokenize_with_subtask("x", st, LONG,
                                          mistake=False, bbox=np.array([296,916,492,1023,1]))
    start = int(np.argmax(lm.astype(int)))
    old = tok[np.clip(start + np.arange(SUBTASK_SLOT), 0, len(tok)-1)]
    old_sup = [int(t) for t, w in zip(old, lm[np.clip(start+np.arange(SUBTASK_SLOT),0,len(tok)-1)]) if w > 0]
    print(f"\n[참고] 예전 방식(start+arange(14)) 감독 토큰 {len(old_sup)}개, EOS {'있음' if EOS in old_sup else '없음'}")
    print(f"       {sp.decode([t for t in old_sup if t != EOS])!r}  <- subtask 뒷부분이 잘린다")
    ok["⑤ 예전 방식은 EOS 를 놓친다(회귀 감시)"] = EOS not in old_sup

    print()
    for k, v in ok.items():
        print(f"  {'OK ' if v else '✗  '} {k}")
    print("\nBBOX-CE-TEST " + ("OK" if all(ok.values()) else "FAIL"))
    if not all(ok.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
