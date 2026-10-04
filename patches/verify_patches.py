#!/usr/bin/env python3
"""소스 패치가 전부 들어가 있는지 **하드 체크** 한다. 하나라도 빠지면 종료 코드 1.

왜 필요한가
  2026-10-04: bootstrap 이 `if [ ! -d "$R" ]` 안에서만 apply_loss_mask.py 를 불러서,
  저장소가 이미 있는 경로로 재구축하면 JAX 쪽 손실 마스킹이 통째로 빠졌다.
  증상이 없다 — 학습은 그냥 돌고, action 손실의 18/32 축이 "0 을 맞혀라" 가 되어
  조용히 희석될 뿐이다. 같은 날 subtask CE 와 subtask 통과도 빠져 있었다.

  학습을 띄우기 전에 이걸 돌릴 것. 패치는 파드가 죽었을 때만 돌아서 평소에
  검증되지 않는다 — 조용히 빠지는 경로를 하나도 두면 안 된다.

사용법
  python3 verify_patches.py <RoboSynChallenge 루트> [--mistake] [--mem]
"""
import pathlib, sys

P = "policy/pi05/src/openpi"

# (파일, 반드시 들어 있어야 하는 문자열, 설명)
CORE = [
    (f"{P}/models/pi0.py",                "_RSC_REAL_ACTION_DIM",      "손실 마스킹 (패딩 18축 제외)"),
    (f"{P}/models_pytorch/pi0_pytorch.py", "_RSC_REAL_ACTION_DIM",     "손실 마스킹 (pytorch 경로)"),
]
SUBTASK = [
    (f"{P}/models/tokenizer.py",          "tokenize_with_subtask",     "subtask 슬롯 토크나이즈"),
    (f"{P}/transforms.py",                "class InjectSubtask",       "frame_index -> subtask 문장"),
    (f"{P}/policies/libero_policy.py",    'inputs["subtask"]',         "subtask 통과 (dict 재생성)"),
    (f"{P}/models/pi0.py",                "RSC_SUBTASK",               "subtask CE 손실"),
    (f"{P}/models/gemma.py",              "RSC_SUBTASK",               "attention 마스크"),
    (f"{P}/training/config.py",           "subtask_sentences",         "데이터 배선"),
    (f"{P}/models/tokenizer.py",          "RSC_SUBTASK_V2",            "Task 제거 · EOS"),
    ("policy/pi05/scripts/train.py",      "RSC_SUBTASK_V2",            "분리 로깅"),
    (f"{P}/training/config.py",           "RSC_DECODE",                "추론 디코드"),
    (f"{P}/models/pi0.py",                "RSC_EOSFIX",                "EOS 뒤 pad (R2)"),
]
MISTAKE = [
    (f"{P}/models/tokenizer.py",          "Mistake:",                  "프롬프트 필드"),
    (f"{P}/transforms.py",                "class InjectMistake",       "episode/frame -> mistake"),
    (f"{P}/transforms.py",                "mistake_field",             "TokenizePrompt 스위치"),
    (f"{P}/policies/libero_policy.py",    'inputs["mistake"]',         "mistake 통과"),
    (f"{P}/training/config.py",           "mistake_starts",            "데이터 배선"),
    (f"{P}/training/config.py",           '"episode_index": "episode_index"', "repack 통과"),
]
MEM = [
    (f"{P}/models/siglip.py",             "SpaceTimeAttention",        "space-time 분리 어텐션"),
    (f"{P}/models/siglip.py",             "spacetime_stride",          "4층마다 temporal"),
    (f"{P}/models/model.py",              "frame_valid",               "프레임 유효 마스크"),
    (f"{P}/models/pi0.py",                "PI05_MEM_FRAMES",           "num_frames 배선"),
    (f"{P}/training/data_loader.py",      "PI05_MEM_FRAMES",           "delta_timestamps"),
    (f"{P}/shared/image_tools.py",        "lead",                      "5D resize"),
    ("policy/pi05/scripts/train.py",      "RSC_MEMLOG",                "첫 배치 이미지 로깅 (R10)"),
]
# 추론 경로. 평가 스크립트가 있을 때만 검사한다.
MEM_INFER = [
    ("policy/pi05/deploy_policy.py",      "RSC_MEMSTEP",               "버퍼 시간축 = env step (R9)"),
    ("policy/pi05/pi_model.py",           "RSC_MEMINFER",              "4D transpose (R8)"),
    (f"{P}/policies/libero_policy.py",    "RSC_MEMINFER",              "추론 frame_valid 통과 (R8)"),
]
ROLLOUT = [
    ("scripts/eval_policy.py",            "RSC_ALIGN",                 "녹화 (o_t, a_t) 정렬 (R1)"),
]


def run(root: pathlib.Path, groups):
    bad = 0
    for title, items in groups:
        print(f"\n[{title}]")
        for rel, needle, why in items:
            p = root / rel
            if not p.exists():
                print(f"  ✗  {why:34} 파일 없음: {rel}")
                bad += 1
                continue
            if needle in p.read_text():
                print(f"  OK {why:34} {rel.split('/')[-1]}")
            else:
                print(f"  ✗  {why:34} {rel}  <- '{needle}' 없음")
                bad += 1
    return bad


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = pathlib.Path(sys.argv[1])
    groups = [("핵심 (항상 필요)", CORE)]
    if "--subtask" in sys.argv or "--all" in sys.argv:
        groups.append(("subtask", SUBTASK))
    if "--mistake" in sys.argv or "--all" in sys.argv:
        groups.append(("mistake", MISTAKE))
    if "--mem" in sys.argv or "--all" in sys.argv:
        groups.append(("MEM", MEM))
        if (root / "policy/pi05/deploy_policy.py").exists():
            groups.append(("MEM 추론", MEM_INFER))
    if (root / "scripts/eval_policy.py").exists() and (
            "--rollout" in sys.argv or "--all" in sys.argv):
        groups.append(("롤아웃 녹화", ROLLOUT))
    bad = run(root, groups)
    print()
    if bad:
        sys.exit(f"VERIFY-PATCHES FAIL — 빠진 패치 {bad} 개. 해당 apply_*.py 를 돌릴 것.")
    print("VERIFY-PATCHES OK")


if __name__ == "__main__":
    main()
