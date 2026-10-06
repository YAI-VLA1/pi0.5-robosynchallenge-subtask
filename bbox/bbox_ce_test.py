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
WBOX_SLOT, WBOX_MID = 4, 2
WBOX_ON = __import__('os').environ.get('PI05_BBOX_WRIST','0')=='1'
EOS, PAD = 1, 0


def gather_idx(start, bbox_on):
    """pi0.py 의 _rsc_subtask_ce 와 **같은** 인덱스를 만든다."""
    if bbox_on and WBOX_ON:
        w0 = BBOX_SLOT + WBOX_MID
        s0 = w0 + WBOX_SLOT + BBOX_MID
        offs = np.concatenate([np.arange(BBOX_SLOT), w0 + np.arange(WBOX_SLOT),
                               s0 + np.arange(SUBTASK_SLOT)])
        return start + offs, BBOX_SLOT + WBOX_SLOT
    if bbox_on:
        offs = np.concatenate([np.arange(BBOX_SLOT),
                               BBOX_SLOT + BBOX_MID + np.arange(SUBTASK_SLOT)])
        return start + offs, BBOX_SLOT
    return start + np.arange(SUBTASK_SLOT), 0


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
    # visible 3상태: 1=실제 박스, 2="박스 없음"(가르친다), 0=추론 빈 슬롯.
    #   라벨에는 0 이 안 들어온다 — 0 을 쓰면 추론 경로를 테스트하게 된다.
    for tag, bb in (("보임", np.array([296, 916, 492, 1023, 1])),
                    ("박스없음", np.array([0, 0, 0, 0, 2]))):
        tok, m, lm = tk.tokenize_with_subtask("x", st, LONG0,
                                              mistake=False, bbox=bb,
                                              wbox=(bb if WBOX_ON else None))
        start, idx, tgt, msk = supervised(tok, lm, True)
        sup = [int(t) for t, w in zip(tgt, msk) if w > 0]
        nloc = sum(1 for t in sup if 256000 <= t <= 257023)
        has_eos = EOS in sup
        print(f"[{tag}] loss mask 첫 True = {start}  ·  감독 토큰 {len(sup)}개 (<loc> {nloc}, EOS {'있음' if has_eos else '없음'})")
        print(f"       {sp.decode([t for t in sup if t != EOS])!r}")
        if tag == "보임":
            ok["① 감독 = bbox(+wrist) + subtask(EOS까지)"] = (nloc == (8 if WBOX_ON else 4) and has_eos)
            ok["② 사이 텍스트는 제외"] = all(
                t in (EOS,) or 256000 <= t <= 257023 or
                t in sp.encode(" " + LONG0) for t in sup)
            start_vis = start
        else:
            ok["③ '박스 없음' 도 슬롯 위치 그대로"] = (start == start_vis)
            ok["④ 박스 없음도 <loc0000> 로 감독"] = (nloc == (8 if WBOX_ON else 4) and has_eos)

    # ⑤ 예전 버그 재현. **긴 문장**이어야 드러난다 — 짧은 문장은 subtask 앞 6칸
    #    안에 EOS 가 들어와 예전 방식도 우연히 통과한다.
    LONG = "open the left gripper to drop the pen in the holder"
    tok, m, lm = tk.tokenize_with_subtask("x", st, LONG,
                                          mistake=False, bbox=np.array([296,916,492,1023,1]),
                                          wbox=(np.array([100,200,300,400,1]) if WBOX_ON else None))
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
