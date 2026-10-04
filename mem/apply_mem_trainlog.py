#!/usr/bin/env python3
"""MEM(T>1) 일 때 train.py 의 첫 배치 이미지 로깅이 죽는 것을 고친다. (리뷰 R10)

무엇이 문제인가
  train.py 는 카메라별 img[i] 를 axis=1 로 이어 wandb.Image 에 넘긴다.
  T=1 이면 (224,224,3) 이라 문제가 없는데, T=6 이면 (6,224,224,3) 이 되고
  이어 붙이면 (6,672,224,3) 이다. wandb 의 NumPy 이미지 경로는 squeeze 후
  PIL.fromarray 를 쓰는데 이 4차원 배열은 이미지로 변환할 수 없다.

  학습 1스텝 테스트(mem_train_step.py)는 데이터·모델만 불러서 이 코드를 안 지난다.
  그래서 실제 train.py 를 돌리기 전까지 안 보인다.

어떻게 고치나
  프레임 축이 있으면 **현재 프레임만**(마지막 칸) 로깅한다. 과거 프레임은
  같은 장면의 1초 전이라 sanity check 에 추가 정보를 주지 않는다.
"""
import pathlib, sys

MARK = "RSC_MEMLOG"
F = "scripts/train.py"
A = '''    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]'''
B = f'''    # {MARK}: MEM 이면 이미지가 (b, T, h, w, c) 다. 현재 프레임만 로깅한다 —
    # 4차원을 그대로 넘기면 wandb 의 PIL 변환에서 죽는다.
    def _cur_frame(x):
        a = np.array(x)
        return a[-1] if a.ndim == 4 else a

    images_to_log = [
        wandb.Image(np.concatenate(
            [_cur_frame(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]'''


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    p = root / F
    s = p.read_text()
    if MARK in s:
        print(f"  {F}: 이미 적용됨")
        return
    n = s.count(A)
    if n != 1:
        raise SystemExit(f"★ {F}: 앵커가 {n} 번 나온다 (1 이어야 함)")
    p.write_text(s.replace(A, B, 1))
    print(f"  {F}: ✓ 적용")


if __name__ == "__main__":
    main()
