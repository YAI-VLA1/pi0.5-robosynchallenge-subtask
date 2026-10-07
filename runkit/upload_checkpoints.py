"""체크포인트가 저장되면 HuggingFace 로 올리고, 오래된 것은 원격에서 지운다.

파드가 유휴 판정으로 정지되면 /workspace 는 통째로 비워지므로, HuggingFace 가
유일한 영속 저장소다. 진행 기록(STATE)은 /root 에 두어 재시작 후에도 이어진다.

보관 정책: 5,000 의 배수(평가용 마일스톤) + 최근 2개. 8.9GB x 40개를 다 쌓으면
356GB 가 되므로 90GB 선에서 묶는다.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

from huggingface_hub import HfApi

CKPT_ROOT = pathlib.Path(os.environ["CKPT_ROOT"])          # .../<config>/<exp>
REPO_ID = os.environ.get("HF_REPO_ID", "yai-robosync/pi05-click-bell-cos40k")
# 런마다 상태 파일이 달라야 한다. 안 그러면 다른 런의 업로드 기록을 물려받는다.
STATE = pathlib.Path(os.environ.get(
    "UPLOAD_STATE", "/root/rsc_recover/uploaded_steps.json"))   # /root = 파드 재시작에 살아남는다
POLL_SECONDS = 60
FINAL_STEP = int(os.environ.get("FINAL_STEP", "39999"))
MILESTONE = int(os.environ.get("MILESTONE", "5000"))  # 이 배수는 평가용으로 원격에 남긴다
# 82k 런에서 5,000 배수면 17개 x 9.5GB = 160GB 다. 10,000 으로 두면 절반.
KEEP_RECENT = 2

api = HfApi(token=pathlib.Path("/root/.cache/huggingface/token").read_text().strip())


def load_state() -> set:
    return set(json.loads(STATE.read_text())) if STATE.exists() else set()


def save_state(done: set) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(sorted(done, key=int), indent=2))


def complete(step_dir: pathlib.Path) -> bool:
    """orbax 가 다 쓴 체크포인트인지 본다. 쓰는 중인 디렉터리를 올리지 않기 위함이다."""
    if not (step_dir / "_CHECKPOINT_METADATA").exists():
        return False
    if not (step_dir / "params").is_dir() or not (step_dir / "train_state").is_dir():
        return False
    try:
        newest = max((p.stat().st_mtime for p in step_dir.rglob("*") if p.is_file()), default=0)
    except OSError:      # orbax 가 max_to_keep 로 스캔 도중 지웠다
        return False
    return (time.time() - newest) > 120


def prune_remote(done: set) -> None:
    """마일스톤과 최근 몇 개만 남긴다. train_state 까지 올리므로 하나가 8.9GB 다."""
    steps = sorted((int(s) for s in done), reverse=True)
    keep = set(steps[:KEEP_RECENT]) | {s for s in steps if s % MILESTONE == 0}
    # 원격에 실제로 남아 있는 것만 지운다. done 은 지금까지 올린 전부라,
    # 이걸로 돌면 이미 지운 step 을 매번 다시 지우려 들어 404 가 쌓인다.
    try:
        present = {int(f.split("/")[1]) for f in api.list_repo_files(REPO_ID, repo_type="model")
                   if f.startswith("checkpoints/") and f.split("/")[1].isdigit()}
    except Exception as exc:
        print(f"원격 목록 조회 실패: {type(exc).__name__} {exc}", flush=True)
        return
    deleted = False
    for s in steps:
        if s in keep or s not in present:
            continue
        try:
            api.delete_folder(f"checkpoints/{s}", repo_id=REPO_ID, repo_type="model",
                              commit_message=f"prune checkpoint {s}")
            print(f"원격 정리: step {s} 삭제", flush=True)
            deleted = True
        except Exception as exc:
            print(f"원격 정리 실패 step {s}: {type(exc).__name__} {exc}", flush=True)

    # 폴더를 지워도 LFS 블롭은 git 히스토리에 남아 용량을 계속 차지한다.
    # 히스토리를 한 커밋으로 압축해야 실제로 회수된다 (현재 파일은 보존).
    # 2026-09-16 02:00 경 이것 때문에 LFS 업로드가 용량 초과로 막혔다.
    if deleted:
        try:
            api.super_squash_history(repo_id=REPO_ID, repo_type="model",
                                     commit_message="squash: reclaim storage from pruned checkpoints")
            print("원격 히스토리 압축 완료", flush=True)
        except Exception as exc:
            print(f"히스토리 압축 실패: {type(exc).__name__} {exc}", flush=True)


def main() -> None:
    print(f"감시: {CKPT_ROOT}\n업로드 대상: {REPO_ID}", flush=True)
    private = os.environ.get("HF_REPO_PRIVATE", "1") not in ("0", "false", "False")
    api.create_repo(REPO_ID, repo_type="model", private=private, exist_ok=True)
    print(f"  저장소 {'비공개' if private else '공개'}", flush=True)

    done = load_state()
    print(f"이미 올린 step: {sorted(done, key=int) or '없음'}", flush=True)

    while True:
        if CKPT_ROOT.is_dir():
            pending = sorted((d for d in CKPT_ROOT.iterdir()
                              if d.is_dir() and d.name.isdigit() and d.name not in done),
                             key=lambda p: int(p.name))

            # 밀렸을 때는 최신 것을 먼저 지킨다. 하나 올리는 데 9.5GB / 약 13분이라,
            # 오래된 것부터 순서대로 올리면 학습이 저만치 앞서가 있는 동안 낡은
            # 체크포인트를 붙들게 된다. 2026-09-16 파드 사망 때 업로더가 세 개나
            # 뒤처져 있어 3,400스텝을 잃었다.
            # 마일스톤(평가용)과 가장 최신 것만 남기고, 중간 것은 건너뛴다.
            if len(pending) > 1:
                newest = max(int(d.name) for d in pending)
                take, skip = [], []
                for d in pending:
                    n = int(d.name)
                    (take if (n == newest or n % MILESTONE == 0) else skip).append(d)
                for d in skip:
                    done.add(d.name)
                    print(f"건너뜀 step {d.name} (밀려서 최신 우선)", flush=True)
                if skip:
                    save_state(done)
                # 최신부터 올린다
                pending = sorted(take, key=lambda p: -int(p.name))
            for step_dir in pending:
              try:
                if not complete(step_dir):
                    continue
                gb = sum(p.stat().st_size for p in step_dir.rglob("*") if p.is_file()) / 1e9
                print(f"업로드 시작 step {step_dir.name} ({gb:.1f} GB)", flush=True)
                try:
                    api.upload_folder(
                        folder_path=str(step_dir),
                        path_in_repo=f"checkpoints/{step_dir.name}",
                        repo_id=REPO_ID,
                        repo_type="model",
                        commit_message=f"checkpoint step {step_dir.name} (40k cosine, corrected norm stats)",
                    )
                except Exception as exc:      # 네트워크 실패는 다음 주기에 다시 시도한다
                    print(f"업로드 실패 step {step_dir.name}: {type(exc).__name__} {exc}", flush=True)
                    continue
                done.add(step_dir.name)
                save_state(done)
                print(f"업로드 완료 step {step_dir.name}", flush=True)
                prune_remote(done)

                if int(step_dir.name) >= FINAL_STEP:
                    print("마지막 체크포인트까지 올렸다. 감시를 끝낸다.", flush=True)
                    return
              except Exception as exc:   # 어떤 이유로도 업로더가 멈추면 안 된다
                  print(f"step {step_dir.name} 처리 중 예외: {type(exc).__name__} {exc}", flush=True)
                  continue
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
