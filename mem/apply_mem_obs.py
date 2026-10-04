import pathlib
p = pathlib.Path("/workspace/rsc_ws/RoboSynChallenge/policy/pi05/src/openpi/models/model.py")
s = p.read_text()
pairs = [
  # images 는 MEM 에서 (B,T,H,W,C) 가 될 수 있다. *b 를 공유하면 image_masks(*b)
  # 와 state(*b s) 가 (4,) 를 요구해 충돌한다 — 별도 변수명 *ib 를 쓴다.
  ('    images: dict[str, at.Float[ArrayT, "*b h w c"]]',
   '    # MEM 일 때 (*b, T, h, w, c). *b 를 image_masks/state 와 공유하면 충돌하므로\n'
   '    # 별도 변수명을 쓴다 (h/w/c 검사는 유지된다).\n'
   '    images: dict[str, at.Float[ArrayT, "*ib h w c"]]'),
  ('            tokenized_prompt=data.get("tokenized_prompt"),',
   '            frame_valid=data.get("frame_valid"),\n'
   '            tokenized_prompt=data.get("tokenized_prompt"),'),
  ('    frame_valid: dict[str, at.Bool[ArrayT, "*b t"]] | None = None',
   '    frame_valid: dict[str, at.Bool[ArrayT, "*fb t"]] | None = None'),
]
if '"*ib h w c"' in s:
    print("이미 적용됨")
else:
    for a, _ in pairs:
        assert s.count(a) == 1, f"앵커 {s.count(a)}개: {a[:50]}"
    for a, n in pairs:
        s = s.replace(a, n, 1)
    p.write_text(s); print("OK — images 주석 분리 + from_dict 에 frame_valid 추가")
