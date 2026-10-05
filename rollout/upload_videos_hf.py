#!/usr/bin/env python3
"""영상 폴더들을 HF 에 올린다. /workspace 는 날아가므로 재생성 비용이 큰 것만."""
import pathlib, sys
from huggingface_hub import HfApi

REPO = "yai-robosync/handover-rollouts"
api = HfApi()
for d, name in [a.split("=", 1) for a in sys.argv[1:]]:
    p = pathlib.Path(d)
    if not p.exists():
        print(f"  건너뜀 (없음) {d}"); continue
    n = len(list(p.rglob("*")))
    print(f"  {d} ({n} 파일) -> {name}", flush=True)
    api.upload_folder(folder_path=d, repo_id=REPO, repo_type="dataset", path_in_repo=name)
print("VIDEO-UPLOAD-DONE")
