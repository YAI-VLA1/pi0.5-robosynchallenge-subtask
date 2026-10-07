"""warm-start 체크포인트(10661)와 obb 라벨을 내려받는다."""
import sys, pathlib, tarfile
from huggingface_hub import snapshot_download, hf_hub_download
tok = open('/root/.cache/huggingface/token').read().strip()
REPO = "yai-robosync/pi05-items-handover-subtask-mistake-mem-bbox-ep1"
dst = pathlib.Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
print("체크포인트 10661 ...", flush=True)
snapshot_download(REPO, allow_patterns=["checkpoints/10661/**"], local_dir=str(dst), token=tok, max_workers=8)
print("obb 라벨 ...", flush=True)
p = hf_hub_download(REPO, "labels/obb_labels.tgz", local_dir=str(dst), token=tok)
out = pathlib.Path("/workspace/obb_labels"); out.mkdir(parents=True, exist_ok=True)
with tarfile.open(p) as t:
    t.extractall(out)
# tgz 안에 obb_labels/ 가 한 겹 더 있다. npy 가 실제로 있는 곳을 out 바로 아래로 올린다.
npys = list(out.rglob("*.npy"))
if npys and npys[0].parent != out:
    for f in npys:
        f.rename(out / f.name)
print("obb 파일 수:", len(list(out.glob("*.npy"))), flush=True)
print("SEED-DONE", flush=True)
