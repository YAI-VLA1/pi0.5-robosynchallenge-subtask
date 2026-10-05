#!/usr/bin/env python3
"""mistake 메타데이터용 TrainConfig 를 config.py 에 넣는다.

mistake_starts 는 병합 데이터셋의 meta/mistake_starts.json 에서 읽어
**리터럴로 박아 넣는다** — config 가 런타임 파일에 의존하지 않도록.
데이터셋을 다시 만들면 이 스크립트를 다시 돌리면 된다.
"""
import json, pathlib, sys

BBOX_LABELS_DIR = "/workspace/bbox_tokens"
NAME     = "pi05_robosyn_items_handover_mistake_cos82k"
NAME_SUB = "pi05_robosyn_items_handover_subtask_mistake_cos82k"
ANCHOR = '''    TrainConfig(
        name="pi05_robosyn_items_handover_lora_cos82k",'''

# subtask 문장은 대본 스케줄(327 프레임)에서 나온다. 롤아웃 에피소드는 길이가
# 128~285 라 경계가 정확히 맞지는 않는다. 다만 실측 t_grasp 중앙값 86 이
# 대본의 'close the right gripper on the pen' 구간 74~89 안에 들어와서,
# 잘라낸 구간(실수 직전 50 + 직후 50)에서는 어긋남이 작다.
SUBTASK_ENDS = (15, 50, 74, 89, 113, 148, 172, 202, 217, 227, 247, 272, 292, 302, 327)
SUBTASK_SENTENCES = (
    'open both grippers', 'move the right arm toward the pen',
    'lower the right gripper onto the pen', 'close the right gripper on the pen',
    'lift the pen with the right arm', 'move the right arm to the handover pose',
    'move the left arm toward the pen', 'bring the left gripper around the pen',
    'close the left gripper on the pen', 'open the right gripper to release the pen',
    'move the right arm back', 'carry the pen with the left arm',
    'move the pen over the holder',
    'open the left gripper to drop the pen in the holder',
    'lift the left arm away from the holder')


def main():
    pi05 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/workspace/rsc_ws/RoboSynChallenge/policy/pi05")
    ds = pi05 / "training_data/RoboSynChallenge/cobotmagic_Sim_items_handover_mix"
    starts = json.loads((ds / "meta/mistake_starts.json").read_text())
    pairs = tuple(sorted((int(k), int(v)) for k, v in starts.items()))
    lit = "(" + ", ".join(f"({a}, {b})" for a, b in pairs) + ")"

    cfg = pi05 / "src/openpi/training/config.py"
    s = cfg.read_text()
    for name, extra in ((NAME, ""),
                        (NAME_SUB,
                         f"\n            subtask_ends={SUBTASK_ENDS},"
                         f"\n            subtask_sentences={SUBTASK_SENTENCES},")):
        block = block_for(name, lit, extra)
        b0, b1 = f"    # <{name}>\n", f"    # </{name}>\n"
        block = b0 + block + b1
        i = s.find(b0)
        if i >= 0:
            # 데이터셋을 다시 만들면 episode 번호와 t_mistake 가 바뀐다. 이름이
            # 있다고 건너뛰면 **이전 라벨이 다른 에피소드에 붙는다** (리뷰 R14).
            j = s.index(b1) + len(b1)
            if s[i:j] == block:
                print(f"  변경 없음: {name}")
                continue
            s = s[:i] + block + s[j:]
            print(f"  갱신: {name} · mistake 에피소드 {len(pairs)}")
        else:
            assert s.count(ANCHOR) == 1, f"앵커가 {s.count(ANCHOR)} 번 나온다"
            s = s.replace(ANCHOR, block + ANCHOR, 1)
            print(f"  추가: {name} · mistake 에피소드 {len(pairs)}")
    cfg.write_text(s)
    print("MISTAKE-CONFIG-DONE")


def block_for(name, lit, extra):
    return f'''    # RSC_MISTAKE: 데모 1,000 + 실패 롤아웃 98 (mistake 라벨 포함) 병합 데이터셋.
    # baseline(pi05_robosyn_items_handover_lora_cos82k) 과 같은 82k cosine 이라
    # 같은 step 끼리 비교할 수 있다. PI05_MISTAKE=1 이어야 프롬프트에 필드가 들어간다.
    TrainConfig(
        name="{name}",
        project_name="robosyn-items-handover-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotEmbodiChainDataConfig(
            repo_id="RoboSynChallenge/cobotmagic_Sim_items_handover_mix",
            # 정규화 통계는 데모 전용 데이터셋 것을 그대로 쓴다. 롤아웃이 4% 라
            # 통계는 사실상 같고, 81999 가 본 입력 분포를 안 바꾸는 편이 재개에 안전하다.
            # asset_id 를 고정하지 않으면 repo_id(_mix) 경로를 찾다가 통계를 못 읽는다.
            # assets 는 DataConfig 가 아니라 DataConfigFactory 의 필드다.
            assets=AssetsConfig(asset_id="RoboSynChallenge/cobotmagic_Sim_items_handover"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            image_key_high="observation.images.cam_high",
            image_key_left="observation.images.cam_left_wrist",
            image_key_right="observation.images.cam_right_wrist",
            state_key="observation.state",
            # RSC_BBOX: 투영 GT 박스 라벨. 경로를 **리터럴로** 박는다 —
            #   변수 이름을 그대로 내보내면 config.py 에서 NameError 다.
            bbox_labels_dir={BBOX_LABELS_DIR!r},
            mistake_starts={lit},{extra}
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=82_000,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=82_000),
        batch_size=4,
        fsdp_devices=1,
        wandb_enabled=True,
        save_interval=1_000,
        keep_period=10_000,
    ),
'''


if __name__ == "__main__":
    main()
