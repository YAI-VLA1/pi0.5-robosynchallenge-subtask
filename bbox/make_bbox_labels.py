#!/usr/bin/env python3
"""회전 박스 라벨 -> PaliGemma 검출 형식(축정렬 4토큰)으로 변환한다.

왜 축정렬인가
  PaliGemma 는 <loc_ymin><loc_xmin><loc_ymax><loc_xmax> **그 순서** 로 검출을
  사전학습했다. 어휘뿐 아니라 **의미와 순서까지** 재사용하려면 그대로 따라야 한다.
  (cx,cy,w,h,theta 5토큰은 3~5번 자리가 '길이·각도' 라는 새 용법이라 재학습이 필요하다.)

각도를 버려도 되는 근거 (실측 84,179 프레임)
  가는 물체의 축정렬 박스는 '물체가 이 박스의 대각선을 따라 놓였다' 는 뜻이다.
  올바른 대각선을 고르면 각도 복원 오차 중앙값 6.1도 (90% 9.2도) 다.
  즉 잃는 것은 각도 전체가 아니라 **어느 대각선이냐 1비트** 이고, 그 1비트는
  모델이 이미지를 보면 풀린다. 박스의 일은 '어디를 볼지 좁히는 것' 이다.

부수 효과
  축정렬 박스는 둥근 물체에도 잘 정의된다 -> 이심률 가중·대칭 판정이 통째로 불필요.
"""
from __future__ import annotations
import argparse, math, pathlib
import numpy as np

W, H = 640, 480
NBIN = 1024

# visible 열은 **3상태** 다.
#   0  추론용 빈 슬롯 (라벨 파일에는 안 나온다 — 모델이 채운다)
#   1  실제 박스
#   2  "박스 없음" — 화면 밖. 이것도 **CE 로 가르친다.**
#      예전에는 pad 로 두고 손실에서 뺐는데, 그러면 그 자리에 무엇을 넣어도
#      벌점이 없다는 걸 15% 학습시킨 셈이다. 실제로 추론에서 BOS/'Sub' 가 샜다.
#      네 칸 모두 0 은 ymax<=ymin 이라 실제 박스로는 절대 안 나오는 값이라 안전하다.
VIS_NONE, VIS_BOX = 2, 1
NO_BOX_TOK = (0, 0, 0, 0)


def aabb_from_obb(cx, cy, w, h, th):
    """회전 박스 -> 축정렬 외접 사각형 (정규화 좌표).

    ★ w 는 W(640) 로, h 는 H(480) 로 정규화돼 있다. 축이 다르므로 정규화 값을
      그대로 회전시키면 안 된다 — **픽셀로 되돌려 계산하고 다시 정규화**한다.
      (예전 코드는 섞어 썼다: 120x20px 펜 60도에서 83x88 px 가 나왔는데
       올바른 값은 77x114 px 다. 세로가 30% 작았다.)
    """
    c, s = math.cos(th), math.sin(th)
    w_px, h_px = w * W, h * H
    ex = (abs(w_px * c) + abs(h_px * s)) / 2 / W
    ey = (abs(w_px * s) + abs(h_px * c)) / 2 / H
    return cx - ex, cy - ey, cx + ex, cy + ey


def quant(v):
    return int(np.clip(round(v * (NBIN - 1)), 0, NBIN - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/obb_labels")
    ap.add_argument("--out", default="/workspace/bbox_tokens")
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    nvis = ntot = 0
    clipped = 0
    for f in sorted(src.glob("ep*.npy")):
        g = np.load(f)
        # [ymin_tok, xmin_tok, ymax_tok, xmax_tok, visible]  <- PaliGemma 순서
        t = np.zeros((len(g), 5), np.int16)
        for i, (cx, cy, w, h, th, vis) in enumerate(g):
            ntot += 1
            if vis == 0:
                t[i] = NO_BOX_TOK + (VIS_NONE,)      # 화면 밖 — "박스 없음" 을 **가르친다**
                continue
            x0, y0, x1, y1 = aabb_from_obb(cx, cy, w, h, th)
            # 화면 밖으로 삐져나간 부분은 자른다. 완전히 밖이면 보이지 않는 것으로 둔다.
            cx0, cy0 = max(x0, 0.0), max(y0, 0.0)
            cx1, cy1 = min(x1, 1.0), min(y1, 1.0)
            if cx1 <= cx0 or cy1 <= cy0:
                t[i] = NO_BOX_TOK + (VIS_NONE,)      # 완전히 화면 밖
                continue
            if (x0 < 0) or (y0 < 0) or (x1 > 1) or (y1 > 1):
                clipped += 1
            t[i] = (quant(cy0), quant(cx0), quant(cy1), quant(cx1), VIS_BOX)
            nvis += 1
        np.save(out / f.name, t)

    print(f"프레임 {ntot:,} · 박스 있는 프레임 {nvis:,} ({nvis/ntot:.1%})")
    print(f"  '박스 없음' 으로 가르치는 프레임 {ntot-nvis:,} ({1-nvis/ntot:.1%})")
    print(f"  화면 경계에서 잘린 박스 {clipped:,} ({clipped/max(nvis,1):.1%})")
    print(f"형식: [ymin, xmin, ymax, xmax, visible]  각 0~{NBIN-1}  (PaliGemma 순서)")
    print(f"-> {out}")
    print("BBOX-TOKENS-DONE")


if __name__ == "__main__":
    main()
