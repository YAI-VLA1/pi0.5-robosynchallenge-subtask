#!/usr/bin/env python3
"""이 런의 최신 체크포인트를 HF 에서 받는다. 없으면 조용히 끝낸다(첫 실행)."""
import pathlib, sys
from huggingface_hub import HfApi, snapshot_download
tok = pathlib.Path('/root/.cache/huggingface/token').read_text().strip()
repo, dst = sys.argv[1], pathlib.Path(sys.argv[2])
api = HfApi(token=tok)
try:
    fs = api.list_repo_files(repo)
except Exception as e:
    print(f"저장소가 아직 없다 ({e.__class__.__name__}) — 0 부터 시작한다"); sys.exit(0)
steps = sorted({int(f.split('/')[1]) for f in fs
                if f.startswith('checkpoints/') and f.split('/')[1].isdigit()})
if not steps:
    print("체크포인트가 없다 — 0 부터 시작한다"); sys.exit(0)
last = steps[-1]
print(f"최신 {last} 내려받는 중 ...", flush=True)
tmp = dst.parent / "_hf"
snapshot_download(repo, allow_patterns=[f"checkpoints/{last}/**"],
                  local_dir=str(tmp), token=tok, max_workers=8)
src = tmp / "checkpoints" / str(last)
dst.mkdir(parents=True, exist_ok=True)
tgt = dst / str(last)
if not tgt.exists():
    src.rename(tgt)
print(f"-> {tgt}")
