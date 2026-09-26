#!/usr/bin/env python3
"""subtask 실험용 데이터 배선 + TrainConfig 삽입 (RSC_SUBTASK_CFG). 멱등.

하는 일
  1. repack 이 `frame_index` 를 통과시키게 한다 (기본 repack 은 나열한 키만 남긴다)
  2. LeRobotEmbodiChainDataConfig 에 subtask 옵션을 추가하고, 켜지면
     EmbodiChainInputs **앞**에 InjectSubtask 를 꽂는다
     (뒤에 꽂으면 안 된다 — EmbodiChainInputs 가 dict 를 새로 만든다)
  3. baseline 과 같은 82k cosine schedule 로 새 TrainConfig 를 넣는다
     (짧게 돌리면 같은 step 의 learning rate 가 달라 비교가 안 된다)

사용:  python3 prepare_subtask_config.py <pi05 루트> <schedule_sim.json> [--check]
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phases import phases          # noqa: E402
from sentences import sentences    # noqa: E402

MARK = "RSC_SUBTASK_CFG"
TASK = "items_handover"
NAME = "pi05_robosyn_items_handover_subtask_cos82k"
BASE = "pi05_robosyn_items_handover_lora_cos82k"   # prepare_handover.py 가 넣은 것

# ── 1. repack 에 frame_index ────────────────────────────────────────────
REPACK_A = '''                        "actions": "action",
                        "prompt": "prompt",'''
REPACK_B = '''                        "actions": "action",
                        "prompt": "prompt",
                        # {mark}: subtask 라벨이 프레임 번호에서 나온다. 여기서 안 넘기면 사라진다.
                        "frame_index": "frame_index",'''.format(mark=MARK)

# ── 2. 데이터 config 에 subtask 옵션 ────────────────────────────────────
OPT_A = '''    extra_delta_transform: bool = False
    action_sequence_keys: Sequence[str] = ("action",)'''
OPT_B = '''    extra_delta_transform: bool = False
    action_sequence_keys: Sequence[str] = ("action",)
    # {mark}: 구간 끝 프레임과 GT 문장. 비어 있으면 subtask 를 아예 안 넣는다.
    subtask_ends: tuple[int, ...] = ()
    subtask_sentences: tuple[str, ...] = ()'''.format(mark=MARK)

WIRE_A = '''                        "frame_index": "frame_index",
                    }
                )
            ]
        )
'''
WIRE_B = WIRE_A + '''
        # {mark}: repack 그룹에 넣는다. Group.push 는 **뒤에** 붙으므로
        # data_transforms 에 넣으면 EmbodiChainInputs 뒤가 되고, 거기서 dict 를 새로
        # 만들며 frame_index 가 이미 사라진 뒤다. repack 은 학습에만 적용되는데
        # (추론에는 적용되지 않는다) subtask 주입도 학습에서만 필요하니 여기가 맞다.
        #
        # 앵커는 바로 위 frame_index 줄이다 — 우리가 넣은 것이라 유일하고 반드시
        # LeRobotEmbodiChainDataConfig 안에 있다. 예전 앵커(data_transforms 블록)는
        # LeRobotLiberoDataConfig 에도 똑같이 있어서 그쪽에 들어갔다 (2026-09-26 두 번째).
        if self.subtask_sentences:
            repack_transform = repack_transform.push(
                inputs=[_transforms.InjectSubtask(ends=self.subtask_ends,
                                                  sentences=self.subtask_sentences)],
            )
'''.replace("{mark}", MARK)



def build_block(sched: pathlib.Path) -> str:
    ph = phases(sched, TASK)
    ends = tuple(p["end"] for p in ph)
    sents = tuple(sentences(TASK, len(ph)))
    return f'''    TrainConfig(
        # {MARK}: subtask 문장 생성 실험. baseline({BASE}) 과 같은 82k cosine 이라
        # 같은 step 의 체크포인트끼리 비교할 수 있다.
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
            image_key_high="observation.images.cam_high",
            image_key_left="observation.images.cam_left_wrist",
            image_key_right="observation.images.cam_right_wrist",
            state_key="observation.state",
            subtask_ends={ends!r},
            subtask_sentences={sents!r},
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


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    sched = pathlib.Path(sys.argv[2])
    check = "--check" in sys.argv
    cfg = root / "src/openpi/training/config.py"
    s = cfg.read_text()

    if MARK in s:
        print(f"{cfg}: 이미 적용됨")
        return
    anchor = f'        name="{BASE}",'
    # WIRE_A 는 REPACK 적용 후에야 생기므로 여기서 검사하지 않는다.
    missing = [n for n, a in (("repack", REPACK_A), ("opt", OPT_A),
                              ("config 앵커", anchor)) if a not in s]
    if missing:
        sys.exit(f"✗ 앵커 없음: {missing} — prepare_handover.py 가 먼저 돌아야 한다")
    if check:
        print("미적용 (적용 가능)")
        sys.exit(1)

    # 순서 중요: REPACK 이 먼저 들어가야 WIRE 앵커(frame_index 줄)가 생긴다.
    for name, a, b in (("REPACK", REPACK_A, REPACK_B), ("OPT", OPT_A, OPT_B)):
        if s.count(a) != 1:
            sys.exit(f"✗ {name} 앵커가 {s.count(a)}곳 — 더 좁혀야 한다")
        s = s.replace(a, b, 1)
    if s.count(WIRE_A) != 1:
        sys.exit(f"✗ WIRE 앵커가 {s.count(WIRE_A)}곳 — 더 좁혀야 한다")
    s = s.replace(WIRE_A, WIRE_B, 1)

    block = build_block(sched)
    i = s.index(anchor)
    j = s.rindex("    TrainConfig(", 0, i)
    s = s[:j] + block + s[j:]
    cfg.write_text(s)
    print(f"{cfg}: ✓ 적용 — config `{NAME}`")

    ph = phases(sched, TASK)
    print(f"\nsubtask {len(ph)}개:")
    for p, t in zip(ph, sentences(TASK, len(ph))):
        print(f"  {p['id']:>2} ~{p['end']:<4} \"{t}\"")


if __name__ == "__main__":
    main()
