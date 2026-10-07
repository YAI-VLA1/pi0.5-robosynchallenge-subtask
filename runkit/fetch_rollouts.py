"""mistake 학습에 쓰는 실패 롤아웃 98개(subtask) 를 받는다."""
import sys, pathlib
from huggingface_hub import snapshot_download
tok = open('/root/.cache/huggingface/token').read().strip()
dst = pathlib.Path("/workspace/rollout_ds"); dst.mkdir(parents=True, exist_ok=True)
snapshot_download("yai-robosync/handover-rollouts", repo_type="dataset",
                  allow_patterns=["subtask/**"], local_dir=str(dst), token=tok, max_workers=8)
print("ROLLOUT-DONE", len(list((dst/"subtask").rglob("*"))), flush=True)
