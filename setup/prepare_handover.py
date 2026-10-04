#!/usr/bin/env python3
"""items_handover 학습 설정을 openpi config.py 에 넣는다. 멱등하다.

왜 82k 인가
  click_bell 은 74,000 프레임이라 40k 스텝 x batch 4 = 2.16 에포크였다.
  items_handover 는 **327,000 프레임**(에피소드당 327, click_bell 은 74)이라
  같은 40k 를 돌리면 0.49 에포크밖에 안 된다. 1 에포크 = 81,750 스텝이다.
  실측 스텝당 1.412초(wandb) 기준 82k = 약 32시간.

  더 길게(176k, 2.16 에포크) 가면 69시간이라 파드 수명을 감안해 접었다.
  82k 코사인으로 걸어두고 중간 체크포인트를 평가하면, 좋으면 그대로 끝까지
  가고 아니면 중간에 끊을 수 있다.

사전 점검 결과 (coverage_audit.py, 2026-09-20)
  · 물체 위치 커버리지 정상 — pen/holder 모두 샘플링 범위 끝까지 시연이 있다.
    click_bell 처럼 잘린 구간이 없으므로 데이터가 만드는 상한이 없다.
  · norm_stats 손댈 것 없음 — 14축 전부 고유값 95,000 이상, 그리퍼도 0~1 로
    실제 여닫는다. **click_bell 의 그리퍼 보정을 절대 가져오면 안 된다.**
  · delta 공간(dim 0-5)도 밴드 2.6e-03 ~ 7.1e-03 으로 정상.
"""
from __future__ import annotations

import pathlib
import sys

PI05 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
CFG = PI05 / "src/openpi/training/config.py"
NAME = "pi05_robosyn_items_handover_lora_cos82k"

BLOCK = f'''    TrainConfig(
        name="{NAME}",
        project_name="robosyn-items-handover-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotEmbodiChainDataConfig(
            repo_id="RoboSynChallenge/cobotmagic_Sim_items_handover",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            # 이 데이터셋은 표준 LeRobot v2.1 규약을 쓴다 (click_bell 과 다르다).
            image_key_high="observation.images.cam_high",
            image_key_left="observation.images.cam_left_wrist",
            image_key_right="observation.images.cam_right_wrist",
            state_key="observation.state",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # 327,000 프레임이라 82,000 스텝 x batch 4 = 1.00 에포크다.
        # click_bell(74,000 프레임)의 40k 와 스텝 수는 두 배지만 에포크로는 절반이다.
        num_train_steps=82_000,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=82_000),
        batch_size=4,
        fsdp_devices=1,
        wandb_enabled=True,
        # 파드가 죽어도 최악 손실이 ~24분이 되게 1,000 마다 저장한다.
        # keep_period 는 10,000 — 5,000 이면 원격에 17개 x 9.5GB = 160GB 가 쌓인다.
        save_interval=1_000,
        keep_period=10_000,
    ),
'''


REPACK_OLD = """        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "cam_high.color",
                        "observation/left_wrist_image": "cam_left_wrist.color",
                        "observation/right_wrist_image": "cam_right_wrist.color",
                        "observation/state": "observation.qpos",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )"""

REPACK_NEW = """        # 데이터셋마다 LeRobot 규약이 다르다.
        #   click_bell      cam_high.color / observation.qpos        (구식)
        #   items_handover  observation.images.cam_high / observation.state  (표준 v2.1)
        # 기본값은 click_bell 이라 기존 설정은 그대로 동작한다.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": self.image_key_high,
                        "observation/left_wrist_image": self.image_key_left,
                        "observation/right_wrist_image": self.image_key_right,
                        "observation/state": self.state_key,
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )"""

FIELDS_OLD = """    extra_delta_transform: bool = False
    action_sequence_keys: Sequence[str] = ("action",)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,"""

FIELDS_NEW = """    extra_delta_transform: bool = False
    action_sequence_keys: Sequence[str] = ("action",)
    # 데이터셋 컬럼 이름. 기본값 = click_bell 규약.
    image_key_high: str = "cam_high.color"
    image_key_left: str = "cam_left_wrist.color"
    image_key_right: str = "cam_right_wrist.color"
    state_key: str = "observation.qpos"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,"""


def patch_keys(t: str) -> str:
    """LeRobotEmbodiChainDataConfig 의 컬럼 이름을 파라미터로 뺀다."""
    if "image_key_high" in t:
        print("  컬럼 파라미터화 이미 적용됨")
        return t
    for old, new, what in ((FIELDS_OLD, FIELDS_NEW, "필드"),
                           (REPACK_OLD, REPACK_NEW, "repack")):
        if t.count(old) != 1:
            print(f"  {what} 삽입 지점을 찾지 못했습니다 ({t.count(old)}개 일치)")
            raise SystemExit(1)
        t = t.replace(old, new)
    print("  컬럼 파라미터화 적용")
    return t


def main() -> None:
    t = CFG.read_text()
    t2 = patch_keys(t)
    if t2 != t:
        CFG.write_text(t2)
        t = t2
    if NAME in t:
        print("이미 있음")
        return
    anchor = '        name="pi05_robosyn_click_bell_lora_cos40k",'
    if anchor not in t:
        print("삽입 기준점을 찾지 못했습니다 (cos40k 설정이 없음).")
        raise SystemExit(1)
    # cos40k 블록이 끝나는 지점 뒤에 넣는다.
    i = t.index(anchor)
    j = t.index("\n    ),\n", i) + len("\n    ),\n")
    CFG.write_text(t[:j] + BLOCK + t[j:])
    print(f"{NAME} 삽입")


if __name__ == "__main__":
    main()
