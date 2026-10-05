#!/usr/bin/env python3
"""mem-ep1 런의 체크포인트에서 **params 만** 받는다 (평가에는 train_state 가 필요 없다).

  python3 fetch_mem_ep1.py <step> [<step> ...]
"""
import pathlib, sys, time
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "yai-robosync/pi05-items-handover-subtask-mistake-mem-ep1"
OUT  = pathlib.Path("/workspace/mem_ep1")
TOK  = pathlib.Path("/root/.cache/huggingface/token").read_text().strip()


def main():
    steps = sys.argv[1:] or ["7000"]
    allf = list_repo_files(REPO, token=TOK)
    for st in steps:
        want = [f for f in allf
                if f.startswith(f"checkpoints/{st}/")
                and ("/params/" in f or "/assets/" in f or f.endswith("_CHECKPOINT_METADATA"))]
        print(f"step {st}: 파일 {len(want)}", flush=True)
        t0 = time.time()
        for k in range(20):                       # HF 가 IncompleteRead 로 자주 끊긴다
            try:
                for f in want:
                    hf_hub_download(REPO, f, local_dir=str(OUT), token=TOK)
                break
            except Exception as e:
                print(f"  재시도 {k}: {type(e).__name__}", flush=True)
                time.sleep(5)
        else:
            raise SystemExit(f"★ step {st} 회수 실패")
        d = OUT / "checkpoints" / st
        sz = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        print(f"  완료 {sz/1e9:.2f} GB · {time.time()-t0:.0f}s -> {d}", flush=True)
    print("FETCH-DONE")


if __name__ == "__main__":
    main()
