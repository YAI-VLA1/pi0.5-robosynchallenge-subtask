#!/usr/bin/env python3
"""bbox 배선의 구조적 결함을 고친다 (외부 리뷰 ②③).

무엇이 문제였나
  token_loss_mask 하나가 **세 가지 일**을 겸하고 있었다.
    (a) attention causal 블록 경계        <- 라벨 유무와 무관해야 한다
    (b) CE 슬롯 위치 (argmax 로 시작점)     <- 라벨 유무와 무관해야 한다
    (c) 어느 토큰을 감독할지                <- 라벨 유무에 따라 달라져야 한다

  그래서 bbox 가 안 보이는 프레임(15%)과 **추론 전체**(GT 가 없다)에서
    · argmax 가 bbox 시작이 아니라 **subtask 시작**을 돌려주고
    · 디코더가 거기부터 bbox 4개를 써서 **subtask 앞부분을 덮어쓰고**
    · 8칸 뒤부터 subtask 를 생성해 **tail 까지 침범**했다
  attention 구조까지 bbox 가시성에 따라 달라졌다.

  그리고 CE 는 `start + arange(14)` 만 읽어서, bbox 를 켜면
    bbox 4 | 사이 텍스트 4 | subtask 앞 6  까지만 보고
    **subtask 뒤 8칸(EOS 포함)이 통째로 학습에서 빠졌다.**

어떻게 고치나
  (a)(b) loss mask 가 **항상 두 슬롯을 덮는다** — 가시성과 무관하다.
  (c)    안 보이는 bbox 는 슬롯을 pad_id 로 채운다. CE 에서 `tgt == pad_id` 인
         자리를 빼면 감독만 빠지고 구조는 그대로다.
  CE 는 두 슬롯의 인덱스를 **따로** 만든다. EOS 절단은 subtask 구간에만 건다.
"""
import pathlib, sys

MARK = "RSC_BBOXFIX"

# ── ① tokenizer: loss mask 가 가시성과 무관하게 두 슬롯을 덮는다 ───────────
TOK = "src/openpi/models/tokenizer.py"
TOK_A = '''        # RSC_BBOX: 두 슬롯을 모두 덮는다. bbox 가 안 보이면 그 슬롯은 빼야 하므로
        #   bbox_body 가 패딩인 경우를 따로 판정한다.
        bbox_on = bool(bbox_body) and any(t != pad_id0 for t in bbox_body)
        loss = [(lo <= i < lo + self.SUBTASK_SLOT)
                or (bbox_on and lo_bbox <= i < lo_bbox + len(bbox_body))
                for i in range(self._max_len)]'''
TOK_B = f'''        # {MARK}: loss mask 는 **가시성과 무관하게** 두 슬롯을 덮는다.
        #   이 마스크는 attention causal 블록 경계와 디코드 시작점으로도 쓰인다.
        #   라벨이 없다고 그게 달라지면 추론에서 슬롯 위치가 밀린다 (리뷰 ③).
        #   감독 여부는 CE 쪽에서 `tgt == pad_id` 로 가린다.
        loss = [(lo <= i < lo + self.SUBTASK_SLOT)
                or (lo_bbox <= i < lo_bbox + len(bbox_body))
                for i in range(self._max_len)]'''

# ── ② CE: 두 슬롯 인덱스를 따로 만든다 ─────────────────────────────────────
PI0 = "src/openpi/models/pi0.py"
PI0_A = '''        tok = observation.tokenized_prompt
        lm = observation.token_loss_mask
        n_img = prefix_out.shape[1] - tok.shape[1]
        start = jnp.argmax(lm.astype(jnp.int32), axis=-1)               # (b,)
        idx = start[:, None] + jnp.arange(_RSC_SUBTASK_SLOT)            # (b, L)
        idx = jnp.clip(idx, 0, tok.shape[1] - 1)
        pre = jnp.take_along_axis(prefix_out, (n_img + idx - 1)[..., None], axis=1)
        logits = self.PaliGemma.llm(pre, method="decode").astype(jnp.float32)
        tgt = jnp.take_along_axis(tok, idx, axis=1)
        msk = jnp.take_along_axis(lm, idx, axis=1).astype(jnp.float32)
        # RSC_SUBTASK_V2: EOS **까지만** 학습한다. 그 뒤 패딩은 자명해서 손실을 희석시킬 뿐이다.
        # token_loss_mask 는 슬롯 전체를 덮는다 — attention 블록 경계로도 쓰이므로
        # 거기서 줄이면 causal 구조가 깨진다. 그래서 여기서 따로 자른다.
        eos = (tgt == _RSC_EOS_ID).astype(jnp.int32)
        after_eos = jnp.cumsum(eos, axis=-1) - eos          # EOS 다음 자리부터 1
        msk = msk * (after_eos == 0).astype(jnp.float32)'''
PI0_B = f'''        tok = observation.tokenized_prompt
        lm = observation.token_loss_mask
        n_img = prefix_out.shape[1] - tok.shape[1]
        start = jnp.argmax(lm.astype(jnp.int32), axis=-1)               # (b,) 첫 슬롯 시작

        # {MARK}: bbox 를 켜면 시퀀스가  bbox 4 | "; Subtask:" 4 | subtask 14  다.
        #   `start + arange(14)` 로는 **subtask 뒤 8칸(EOS 포함)이 빠진다** (리뷰 ②).
        #   두 슬롯의 인덱스를 따로 만들어 이어 붙인다.
        if _RSC_BBOX_ON:
            off_b = jnp.arange(_RSC_BBOX_SLOT)
            off_s = _RSC_BBOX_SLOT + _RSC_BBOX_MID + jnp.arange(_RSC_SUBTASK_SLOT)
            offs = jnp.concatenate([off_b, off_s])
            n_bbox = _RSC_BBOX_SLOT
        else:
            offs = jnp.arange(_RSC_SUBTASK_SLOT)
            n_bbox = 0
        idx = start[:, None] + offs[None, :]                            # (b, L)
        idx = jnp.clip(idx, 0, tok.shape[1] - 1)
        pre = jnp.take_along_axis(prefix_out, (n_img + idx - 1)[..., None], axis=1)
        logits = self.PaliGemma.llm(pre, method="decode").astype(jnp.float32)
        tgt = jnp.take_along_axis(tok, idx, axis=1)
        msk = jnp.take_along_axis(lm, idx, axis=1).astype(jnp.float32)

        # RSC_SUBTASK_V2: EOS **까지만** 학습한다. 그 뒤 패딩은 자명해서 손실을 희석시킬 뿐이다.
        # token_loss_mask 는 슬롯 전체를 덮는다 — attention 블록 경계로도 쓰이므로
        # 거기서 줄이면 causal 구조가 깨진다. 그래서 여기서 따로 자른다.
        # {MARK}: EOS 절단은 **subtask 구간에만** 건다. bbox 슬롯에는 EOS 가 없다.
        in_sub = (jnp.arange(tgt.shape[1]) >= n_bbox).astype(jnp.int32)[None, :]
        eos = ((tgt == _RSC_EOS_ID) & (in_sub > 0)).astype(jnp.int32)
        after_eos = jnp.cumsum(eos, axis=-1) - eos          # EOS 다음 자리부터 1
        msk = msk * ((after_eos == 0) | (in_sub == 0)).astype(jnp.float32)
        # {MARK}: 박스가 안 보이는 프레임은 슬롯이 pad 다 — 감독에서만 뺀다
        #   (loss mask 는 구조용이라 그대로 둔다).
        if _RSC_BBOX_ON:
            is_bbox = (jnp.arange(tgt.shape[1]) < n_bbox)[None, :]
            msk = msk * (~(is_bbox & (tgt == _RSC_PAD_ID))).astype(jnp.float32)'''

# acc_first 는 이제 bbox 첫 토큰이 된다 — 의미가 바뀌므로 subtask 첫 토큰으로 고정한다
PI0_A2 = '''        acc_first = jnp.mean(hit[:, 0])'''
PI0_B2 = f'''        # {MARK}: bbox 를 켜면 0번은 bbox 첫 토큰이다. 지표 의미를 유지하려고
        #   subtask 슬롯의 첫 자리를 가리킨다.
        acc_first = jnp.mean(hit[:, n_bbox])'''


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    done = 0
    for rel, pairs in ((TOK, [(TOK_A, TOK_B)]), (PI0, [(PI0_A, PI0_B), (PI0_A2, PI0_B2)])):
        p = root / rel
        s = p.read_text()
        if MARK in s:
            print(f"  {rel}: 이미 적용됨"); continue
        for a, _ in pairs:
            n = s.count(a)
            if n != 1:
                raise SystemExit(f"★ {rel}: 앵커가 {n} 번 (1 이어야 함)\n{a.splitlines()[0][:90]}")
        for a, b in pairs:
            s = s.replace(a, b, 1)
        p.write_text(s)
        print(f"  {rel}: ✓ 적용")
        done += 1
    print(f"BBOX-FIX-DONE {done}")


if __name__ == "__main__":
    main()
