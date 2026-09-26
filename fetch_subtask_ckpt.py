#!/usr/bin/env python3
"""HF 에서 최신 subtask 체크포인트를 회수한다.

exit code
  0  받았거나, 저장소에 체크포인트가 하나도 없다(= 처음부터 학습하는 게 맞다)
  1  조회/다운로드 실패 — **반드시 멈춰야 한다**

왜 별도 파일인가
  bootstrap 안에 heredoc 으로 파이썬을 끼워 넣었더니 (a) 시스템 python3 이 불려
  huggingface_hub 이 없었고 (b) 실패해도 그냥 넘어가 step 0 부터 다시 돌았다.
  2026-09-26 복구 중 실제로 그렇게 됐다. 파일로 빼고 `|| exit 1` 을 건다.

사용:  <venv python> fetch_subtask_ckpt.py <repo_id> <ckpt_root>
"""
import os
import pathlib
import re
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download


def main() -> int:
    repo, root = sys.argv[1], pathlib.Path(sys.argv[2])
    token = os.environ.get("HF_TOKEN")
    try:
        files = HfApi(token=token).list_repo_files(repo)
    except Exception as e:                                   # noqa: BLE001
        print(f"  저장소 조회 실패: {type(e).__name__} {e}")
        return 1
    steps = sorted({int(m.group(1)) for f in files if (m := re.search(r"checkpoints/(\d+)/", f))})
    if not steps:
        print("  체크포인트 없음 — 처음부터 학습한다")
        return 0
    last = steps[-1]
    print(f"  최신 {last} 회수 (저장소에 있는 것: {steps})", flush=True)

    root.mkdir(parents=True, exist_ok=True)
    stage = root / "_hf"
    for k in range(30):
        for q in stage.rglob("*"):
            if q.is_file() and q.stat().st_size == 0:
                q.unlink()
        try:
            snapshot_download(repo, allow_patterns=[f"checkpoints/{last}/*"],
                              local_dir=str(stage), max_workers=8, token=token)
            src = stage / "checkpoints" / str(last)
            n = sum(1 for x in src.rglob("*") if x.is_file())
            if n == 0:
                raise RuntimeError("받은 파일이 0개다")
            tgt = root / str(last)
            if tgt.exists():
                shutil.rmtree(tgt)
            shutil.move(str(src), str(tgt))
            shutil.rmtree(stage, ignore_errors=True)
            print(f"  완료: {tgt} ({n} 파일)")
            return 0
        except Exception as e:                               # noqa: BLE001
            print(f"  재시도 {k}: {type(e).__name__} {str(e)[:110]}", flush=True)
    print("  30회 재시도 실패")
    return 1


if __name__ == "__main__":
    sys.exit(main())
