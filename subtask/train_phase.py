#!/usr/bin/env python3
"""관측(영상 3뷰)만으로 위상을 맞히는 분류기.

목적은 성능이 아니라 **판별**이다:
  데모에서 위상 라벨은 timestep 의 결정적 함수다(스크립트가 타이밍을 고정한다).
  그래서 데모 정확도만으로는 "장면을 보고 맞혔는지"와 "시계를 외웠는지"를 못 가른다.
  이 분류기는 **시간 입력을 아예 받지 않는다**. 이미지 3장이 전부다.
  그러므로 데모 홀드아웃 정확도가 높으면 위상은 관측에서 읽힌다는 뜻이고,
  그 다음 질문은 "평가 롤아웃(분포 밖)에서도 읽히는가"가 된다.

입력  (112, 336, 3) — cam_high | cam_right_wrist | cam_left_wrist 가로 결합
모델  ImageNet 사전학습 ResNet18, fc 를 위상 수로 교체
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from phases import frame_labels, phases

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Frames(torch.utils.data.Dataset):
    def __init__(self, files: list[pathlib.Path], ph: list[dict]):
        self.items = []
        self.arrs = {}
        for f in files:
            a = np.load(f, mmap_mode="r")
            self.arrs[f] = a
            lab = frame_labels(ph, len(a))
            self.items += [(f, t, lab[t]) for t in range(len(a))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, t, y = self.items[i]
        x = torch.from_numpy(np.ascontiguousarray(self.arrs[f][t])).permute(2, 0, 1)
        return x, y


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x.float() / 255.0 - MEAN.to(x.device)) / STD.to(x.device)


def build(n_cls: int) -> nn.Module:
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--task", default="items_handover")
    ap.add_argument("--val-episodes", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ph = phases(pathlib.Path(a.schedule), a.task)
    files = sorted(pathlib.Path(a.frames).glob("ep*.npy"))
    if len(files) < a.val_episodes + 10:
        raise SystemExit(f"에피소드가 {len(files)}개뿐 — 너무 적다")
    tr_f, va_f = files[:-a.val_episodes], files[-a.val_episodes:]
    tr, va = Frames(tr_f, ph), Frames(va_f, ph)
    print(f"위상 {len(ph)}개 · 학습 {len(tr_f)}ep/{len(tr)}프레임 · 검증 {len(va_f)}ep/{len(va)}프레임",
          flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build(len(ph)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=6,
                                     pin_memory=True, drop_last=True, persistent_workers=True)
    vl = torch.utils.data.DataLoader(va, batch_size=256, shuffle=False, num_workers=6,
                                     pin_memory=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl))
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    lossf = nn.CrossEntropyLoss()

    for ep in range(a.epochs):
        m.train(); t0 = time.time(); run = 0.0
        for i, (x, y) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                loss = lossf(m(normalize(x)), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if i % 100 == 0:
                print(f"  ep{ep} {i}/{len(dl)} loss {run / (i + 1):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        m.eval(); C = np.zeros((len(ph), len(ph)), int)
        with torch.no_grad():
            for x, y in vl:
                p = m(normalize(x.to(dev))).argmax(1).cpu().numpy()
                for t, q in zip(y.numpy(), p):
                    C[t, q] += 1
        acc = C.trace() / C.sum()
        # 인접 위상까지 맞다고 치는 완화 정확도 — 경계 프레임은 원래 애매하다
        near = sum(C[i, j] for i in range(len(ph)) for j in range(len(ph)) if abs(i - j) <= 1) / C.sum()
        print(f"ep{ep} 검증 정확도 {acc:.4f} · ±1 위상 허용 {near:.4f}", flush=True)

    torch.save({"state": m.state_dict(), "phases": ph, "task": a.task}, a.out)
    per = (C.diagonal() / np.maximum(C.sum(1), 1))
    print("\n위상별 재현율")
    for p, r, n in zip(ph, per, C.sum(1)):
        print(f"  {p['id']:>2} {p['name'][:46]:46s} {r:.3f}  (n={n})")
    json.dump({"acc": float(acc), "near": float(near),
               "per_phase_recall": per.tolist(), "support": C.sum(1).tolist(),
               "confusion": C.tolist(), "phases": ph},
              open(str(a.out) + ".val.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n저장 {a.out}")


if __name__ == "__main__":
    main()
