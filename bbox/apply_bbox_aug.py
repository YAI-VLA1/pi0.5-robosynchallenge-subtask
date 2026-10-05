#!/usr/bin/env python3
"""bbox 를 켜면 **기하 증강을 끈다**. (외부 리뷰)

무엇이 문제인가
  학습 증강에 cam_high 한정으로 기하 변환이 걸려 있다.
      RandomCrop(608x456) -> Resize(640x480) -> Rotate(-5,+5)
  bbox 라벨은 **원본 좌표** 기준인데 이미지만 변형되므로 감독이 어긋난다.
  어긋남의 크기가 우리가 넘어야 할 선보다 크다:
      크롭 오프셋      최대 x 32 px, y 24 px (매 샘플 랜덤)
      회전 ±5도        최대 35 px
      ------------------------------------------------
      state-only 기준선 26.2 px          <- 증강 노이즈가 이걸 넘는다
      펜 박스 긴 변     136.5 px
  하필 기하 증강이 cam_high 에만 걸리는데, 우리 라벨이 바로 그 카메라 것이다.

왜 라벨을 같이 변환하지 않나
  augmax 가 이미지에 적용하는 변환의 역사상을 꺼내 박스에 다시 적용해야 하는데,
  같은 RNG 를 공유해야 하고 Rotate 후 축정렬 박스는 다시 외접을 잡아야 해서
  조용히 틀리기 쉽다. 첫 실험에서는 **끄고** 감독 일관성부터 확인한다.

  색 증강(ColorJitter)은 기하와 무관하므로 그대로 둔다.

켜고 끄기: PI05_BBOX=1 이면 자동으로 꺼진다. PI05_GEOM_AUG=1 로 강제할 수 있다.
"""
import pathlib, sys

MARK = "RSC_BBOXAUG"
F = "src/openpi/models/model.py"

A = '''            transforms = []
            if "wrist" not in key:'''
B = f'''            transforms = []
            # {MARK}: bbox 라벨은 **원본 좌표** 기준이라 기하 증강과 같이 못 쓴다.
            #   크롭 32px + 회전 35px 로 state-only 기준선(26.2px)을 넘는 잡음이 된다.
            #   색 증강은 기하와 무관하므로 남긴다.
            _geom = (os.environ.get("PI05_BBOX", "0") != "1"
                     or os.environ.get("PI05_GEOM_AUG", "0") == "1")
            if "wrist" not in key and _geom:'''


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
        raise SystemExit(f"★ {F}: 앵커가 {n} 번 (1 이어야 함)")
    s = s.replace(A, B, 1)
    if "\nimport os\n" not in s:
        s = s.replace("import logging", "import logging\nimport os", 1)
    p.write_text(s)
    print(f"  {F}: ✓ 적용")


if __name__ == "__main__":
    main()
