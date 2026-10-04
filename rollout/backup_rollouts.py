#!/usr/bin/env python3
"""수집한 롤아웃 데이터셋을 HuggingFace 에 올린다.

왜 필요한가
  2026-10-04 에 baseline 100 에피소드(34,900 프레임, 약 4시간 수집)를
  /workspace 가 날아가며 통째로 잃었다. 영상만 올리고 parquet/meta 는 안 올렸다.
  수집이 끝나는 즉시 이걸 돌린다.
"""
import pathlib, sys
from huggingface_hub import HfApi

src = pathlib.Path(sys.argv[1])
repo = sys.argv[2] if len(sys.argv) > 2 else "yai-robosync/handover-rollouts"
name = sys.argv[3] if len(sys.argv) > 3 else src.name

api = HfApi()
api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
npq = len(list(src.glob("data/chunk-*/*.parquet")))
nmp = len(list(src.glob("videos/**/*.mp4")))
print(f"{src} -> {repo}/{name}  (parquet {npq}, mp4 {nmp})", flush=True)
api.upload_folder(folder_path=str(src), repo_id=repo, repo_type="dataset",
                  path_in_repo=name, ignore_patterns=["*.tmp.*"])
print(f"완료 https://huggingface.co/datasets/{repo}/tree/main/{name}")
