#!/usr/bin/env python3
"""items_handover 데이터셋(8.5GB)을 받는다. 끊기면 0바이트 잔재를 지우고 재시도한다.

HF 가 IncompleteRead 로 자주 끊긴다. 한 번 끊겨서 스크립트가 죽으면 bootstrap
전체가 멈추므로 여기서 완주할 때까지 돈다.

완료 판정은 parquet 1000 + mp4 3000 이다. meta/info.json 은 맨 먼저 도착해서
2.5GB(영상 580/3000)에서 완료로 착각하게 만든다.
"""
import os, pathlib, sys, time
from huggingface_hub import snapshot_download

REPO = "RoboSynChallenge/cobotmagic_Sim_items_handover"
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["DATA"])


def counts():
    return (len(list((root / "data").rglob("*.parquet"))) if (root / "data").exists() else 0,
            len(list((root / "videos").rglob("*.mp4"))) if (root / "videos").exists() else 0)


def main():
    root.mkdir(parents=True, exist_ok=True)
    for k in range(40):
        npq, nmp4 = counts()
        if npq >= 1000 and nmp4 >= 3000:
            print(f"데이터 완료 (parquet {npq}, mp4 {nmp4})")
            return
        for p in root.rglob("*"):
            if p.is_file() and p.stat().st_size == 0:
                p.unlink()
        try:
            snapshot_download(REPO, repo_type="dataset", local_dir=str(root), max_workers=8)
        except Exception as e:
            print(f"  재시도 {k}: {type(e).__name__}", flush=True)
            time.sleep(5)
    npq, nmp4 = counts()
    raise SystemExit(f"40회 재시도 실패 (parquet {npq}/1000, mp4 {nmp4}/3000)")


if __name__ == "__main__":
    main()
