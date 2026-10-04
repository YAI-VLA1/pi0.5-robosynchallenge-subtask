#!/usr/bin/env python3
"""클론한 저장소를 이 컨테이너에서 학습 가능한 상태로 만든다.

1. 시뮬레이터 전용 의존성 3종 제거 — pymeshlab 이 glibc 2.35 휠만 내서
   여기(glibc 2.31)서는 uv sync 가 통째로 막힌다. 학습에는 쓰지 않는다.
2. cos40k 학습 설정 추가.
3. 롤링 체크포인트 보관 수를 1 -> 3 으로. 1이면 다음 체크포인트가 써지는 순간
   지워져서, 업로드(8.9GB, 약 13분)가 저장 간격(24분) 안에 못 끝나면 유실된다.
"""
import pathlib
import re
import sys

pi05 = pathlib.Path(sys.argv[1])

# --- 1. pyproject ---
pp = pi05 / "pyproject.toml"
s = pp.read_text()
for pkg in ('dexsim-engine==0.4.3', 'embodichain', 'robosynchallenge'):
    s2 = re.sub(r'\n\s*"%s",' % re.escape(pkg), '', s)
    if s2 == s:
        print(f"경고: {pkg} 를 찾지 못했다", file=sys.stderr)
    s = s2
s = re.sub(r'\n[^\n]*\b(embodichain|robosynchallenge|dexsim-engine)\b[^\n]*=[^\n]*', '', s)
pp.write_text(s)

# --- 2. cos40k 설정 ---
cfg_path = pi05 / "src/openpi/training/config.py"
c = cfg_path.read_text()
if "pi05_robosyn_click_bell_lora_cos40k" not in c:
    anchor = '        keep_period=5_000,\n    ),\n'
    i = c.index('name="pi05_robosyn_click_bell_lora"')
    j = c.index(anchor, i) + len(anchor)
    block = '''    TrainConfig(
        name="pi05_robosyn_click_bell_lora_cos40k",
        project_name="robosyn-click-bell-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotEmbodiChainDataConfig(
            repo_id="RoboSynChallenge/cobotmagic_Sim_click_bell",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # The 20k baseline cut a 30k cosine off at 2/3, so its last steps still ran a
        # high LR. Match decay_steps to num_train_steps so the schedule lands at 2.5e-6.
        num_train_steps=40_000,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=40_000),
        batch_size=4,
        fsdp_devices=1,
        wandb_enabled=True,
        # The pod gets stopped when the connection goes idle (twice in 21h), so save
        # every ~24 min and ship each one to HuggingFace. keep_period pins the
        # multiples of 5,000 for later evaluation; the rest roll over.
        save_interval=1_000,
        keep_period=5_000,
    ),
'''
    cfg_path.write_text(c[:j] + block + c[j:])
    print("cos40k 설정 삽입")
else:
    print("cos40k 설정 이미 있음")

# --- 3. max_to_keep ---
ck = pi05 / "src/openpi/training/checkpoints.py"
t = ck.read_text()
t2 = t.replace("            max_to_keep=1,", "            max_to_keep=3,")
ck.write_text(t2)
print("준비 완료" if t2 != t else "준비 완료 (max_to_keep 이미 조정됨)")
