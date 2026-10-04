#!/usr/bin/env python3
"""기존 체크포인트에서 **가중치만** 이어받아 짧게 더 학습하는 config 를 만든다. (리뷰 R11)

왜 --resume 로는 안 되나
  --resume 은 train_state 를 통째로 복원한다. 82k 짜리 런이 끝난 체크포인트는
  train_state.step 이 82000 이라 `range(start_step, num_train_steps)` 가 비고
  **한 스텝도 안 돈다.** (저장 전에 step 이 1 증가해서 디렉터리 이름 81999 와
  state.step 82000 이 어긋난다.)

  그리고 기존 config 들의 weight_loader 는 전부 pi05_base 를 가리킨다.
  새 experiment 로 띄우면 81999 가 아니라 **base 에서 시작한다.**

무엇을 하나
  weight_loader 를 <ckpt>/params 로 바꾸고, optimizer 와 step 은 새로 초기화하는
  별도 config 를 등록한다. 짧은 warmup + decay 를 새로 건다.
  데이터·프롬프트 계약(norm_stats, asset_id, subtask/mistake 필드)은 원본 그대로 베낀다.

사용법
  python3 prepare_posttrain_config.py <pi05> <원본 config 이름> <ckpt 경로> [스텝수]

  예) python3 prepare_posttrain_config.py $PI05 \\
        pi05_robosyn_items_handover_subtask_mistake_cos82k \\
        /workspace/ckpt_dl/subtask/checkpoints/81999  12000
"""
import pathlib, re, sys


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    pi05 = pathlib.Path(sys.argv[1])
    base = sys.argv[2]
    ckpt = pathlib.Path(sys.argv[3]).resolve()
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 12_000
    name = f"{base}_post{steps // 1000}k"

    params = ckpt / "params"
    if not params.exists():
        sys.exit(f"★ {params} 가 없다. 체크포인트 경로는 .../checkpoints/<step> 이어야 한다.")

    cfg = pi05 / "src/openpi/training/config.py"
    s = cfg.read_text()
    if name in s:
        print(f"  이미 있음: {name}")
        print("POSTTRAIN-CONFIG-DONE")
        return

    # 원본 TrainConfig 블록을 통째로 베낀다.
    anchor = f'    TrainConfig(\n        name="{base}",'
    if s.count(anchor) != 1:
        sys.exit(f"★ 원본 config '{base}' 를 {s.count(anchor)} 번 찾았다 (1 이어야 함)")
    # 통계는 체크포인트가 쓰던 asset_id 를 그대로 쓴다. 병합(_mix) 데이터셋을 쓰더라도
    # 통계 경로는 데모 데이터셋 이름이다 (prepare_mistake_config 와 같은 선택).
    asset_id = "RoboSynChallenge/cobotmagic_Sim_items_handover"
    i = s.index(anchor)
    j = s.find("\n    TrainConfig(", i + 1)
    j = len(s) if j < 0 else j
    block = s[i:j]

    # ① 이름 ② 가중치 출처 ③ 스텝 수 ④ 스케줄 — 네 군데만 바꾼다.
    new = block.replace(f'name="{base}"', f'name="{name}"', 1)

    wl = re.search(r'        weight_loader=weight_loaders\.CheckpointWeightLoader\([^)]*\),\n', new)
    if not wl:
        sys.exit("★ weight_loader 줄을 못 찾았다")
    new = new.replace(wl.group(0),
                      f'        # RSC_POSTTRAIN: base 가 아니라 기존 체크포인트에서 가중치만 받는다.\n'
                      f'        #   optimizer 와 step 은 새로 초기화된다 (--resume 과 다르다).\n'
                      f'        weight_loader=weight_loaders.CheckpointWeightLoader(\n'
                      f'            "{params}"),\n', 1)

    # ⑤ 정규화 통계: **그 체크포인트가 쓰던 것**을 그대로 쓴다.
    #    config 이름이 바뀌면 assets_dirs 도 바뀌어 통계를 못 찾는다 (실제로 0개가 나왔다).
    #    체크포인트에 동봉된 assets 를 가리키면 입력 분포가 비트 단위로 보존된다.
    asset_line = (f'            assets=AssetsConfig(assets_dir="{ckpt / "assets"}",\n'
                  f'                                asset_id="{asset_id}"),\n')
    existing = re.search(r'            assets=AssetsConfig\([^)]*\),\n', new)
    if existing:
        new = new.replace(existing.group(0), asset_line, 1)
    else:
        hook = re.search(r'            repo_id="[^"]*",\n', new)
        if not hook:
            sys.exit("★ repo_id 줄을 못 찾았다 — assets 를 끼울 자리가 없다")
        new = new.replace(hook.group(0), hook.group(0) + asset_line, 1)

    new = re.sub(r'        num_train_steps=[0-9_]+,\n',
                 f'        num_train_steps={steps:_},\n', new, count=1)
    new = re.sub(r'        lr_schedule=_optimizer\.CosineDecaySchedule\([^)]*\),\n',
                 f'        lr_schedule=_optimizer.CosineDecaySchedule(\n'
                 f'            warmup_steps={max(200, steps // 20)},\n'
                 f'            peak_lr=1e-5,          # 82k 끝의 decay_lr(2.5e-6) 보다 조금 위\n'
                 f'            decay_steps={steps:_},\n'
                 f'            decay_lr=1e-6),\n', new, count=1)
    new = re.sub(r'        keep_period=[0-9_]+,\n',
                 f'        keep_period={max(1000, steps // 6):_},\n', new, count=1)

    for must in (f'name="{name}"', str(params), f"num_train_steps={steps:_}",
                 'assets=AssetsConfig(assets_dir='):
        assert must in new, f"★ 치환 실패: {must}"

    cfg.write_text(s[:i] + new + s[i:])
    print(f"  {name} 추가")
    print(f"    가중치  {params}")
    print(f"    스텝    {steps:,} (warmup {max(200, steps//20):,}, peak_lr 1e-5 -> 1e-6)")
    print(f"    주의    --resume 을 쓰지 말 것. --overwrite 로 새 experiment 를 띄운다.")
    print("POSTTRAIN-CONFIG-DONE")


if __name__ == "__main__":
    main()
