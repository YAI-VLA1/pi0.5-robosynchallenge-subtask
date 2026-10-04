#!/usr/bin/env python3
"""Grounding DINO-T 를 `.` 구분 다중 물체 프롬프트로 돌리고 한 장으로 모아 본다.

세 방식을 비교한다
  combined   "pen. pen holder."   한 번에 — 표준 사용법
  fixed      "pen. holder."       이름 겹침만 없앤 한 번에
  separate   "pen." / "holder."   물체마다 따로

'pen' 이 'pen holder' 의 부분문자열이라 combined 에서 펜이 사라졌었다.
겹침만 없애면 한 번에 가도 되는지가 핵심이다 (지연이 물체 수만큼 곱해지므로).
"""
import json, pathlib, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

M = "IDEA-Research/grounding-dino-tiny"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TH = 0.25

# uid -> (원래 표현, 겹침 없앤 표현)
P = {
    "pen": ("pen", "pen"), "holder": ("pen holder", "holder"),
    "duck": ("tomato", "tomato"), "drawer": ("drawer", "drawer"),
    "milk": ("bottle", "bottle"), "basket": ("basket", "basket"),
    "cube": ("test tube", "test tube"), "rack": ("rack", "rack"),
    "button": ("bell", "bell"),
    "guijiao1": ("centrifuge tube", "centrifuge tube"),
    "guijiao2": ("centrifuge tube", "centrifuge tube"),
    "bottle": ("bottle", "bottle"), "cup": ("cup", "cup"),
}
SKIP = {"table", "CobotMagic", "drawer_background", "distractor_0", "distractor_1"}
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
BIG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)


def iou(a, b):
    x0,y0,x1,y1 = max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    if x1<=x0 or y1<=y0: return 0.0
    i=(x1-x0)*(y1-y0)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)


def main():
    proc = AutoProcessor.from_pretrained(M)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(M).to(DEV).eval()
    print(f"{M} ({DEV})", flush=True)

    def run(im, text):
        inp = proc(images=im, text=text, return_tensors="pt").to(DEV)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inp)
        if DEV == "cuda": torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        r = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=TH, text_threshold=TH,
            target_sizes=[im.size[::-1]])[0]
        return [(l, s.item(), [float(v) for v in b])
                for l, s, b in zip(r["text_labels"], r["scores"], r["boxes"])], ms

    rows, tiles = [], []
    for d in sorted(map(pathlib.Path, sys.argv[1:])):
        task = d.name.replace("masks_", "")
        mp = d / "cam_high_mask.npy"
        if not mp.exists(): continue
        idmap = json.loads((d / "id_map.json").read_text())
        tg = {u: i for u, i in idmap.items() if u not in SKIP and u in P}
        if not tg: continue
        m = np.load(mp)
        im = Image.open(d / "cam_high_rgb.png").convert("RGB")
        gts = {}
        for u, ids in tg.items():
            sel = np.isin(m, ids)
            if sel.any():
                ys, xs = np.where(sel)
                gts[u] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if not gts: continue

        modes = {}
        for k, idx in (("combined", 0), ("fixed", 1)):
            text = ". ".join(sorted({P[u][idx] for u in gts})) + "."
            preds, ms = run(im, text)
            modes[k] = (text, preds, ms)
        # separate
        preds, tot = [], 0.0
        for w in sorted({P[u][1] for u in gts}):
            pr, ms = run(im, w + "."); preds += pr; tot += ms
        modes["separate"] = ("/".join(sorted({P[u][1] for u in gts})), preds, tot)

        for k, (text, preds, ms) in modes.items():
            for u, gt in gts.items():
                want = P[u][0 if k == "combined" else 1]
                cand = [p for p in preds if want in p[0]]
                best = max((iou(gt, p[2]) for p in cand), default=0.0)
                rows.append({"task": task, "mode": k, "uid": u, "prompt": want,
                             "iou": round(best, 3), "ms": round(ms, 1)})

        # 타일: fixed 모드 결과를 그린다
        text, preds, ms = modes["fixed"]
        vis = im.copy(); dr = ImageDraw.Draw(vis)
        for u, gt in gts.items():
            dr.rectangle(gt, outline="#00E676", width=3)
            dr.text((gt[0]+2, max(0, gt[1]-17)), f"GT {u}", fill="#00E676", font=FONT)
        for l, s, b in preds:
            dr.rectangle(b, outline="#FF3B30", width=3)
            dr.text((b[0]+2, min(460, b[3]+2)), f"{l} {s:.2f}", fill="#FF3B30", font=FONT)
        ious = [r["iou"] for r in rows if r["task"] == task and r["mode"] == "fixed"]
        band = Image.new("RGB", (vis.width, 34), "#111111")
        ImageDraw.Draw(band).text(
            (8, 7), f"{task}   '{text}'   IoU {' '.join(f'{v:.2f}' for v in ious)}   {ms:.0f}ms",
            fill="white", font=FONT)
        t = Image.new("RGB", (vis.width, vis.height + 34), "#111111")
        t.paste(vis, (0, 0)); t.paste(band, (0, vis.height))
        tiles.append(t)
        print(f"  {task}: {text}  IoU {ious}  {ms:.0f}ms", flush=True)

    # 모아보기
    cols = 3
    rowsN = (len(tiles) + cols - 1) // cols
    W, H = tiles[0].size
    sheet = Image.new("RGB", (cols * W + (cols+1)*8, rowsN * H + (rowsN+1)*8 + 40), "#1A1A1A")
    ImageDraw.Draw(sheet).text((12, 12), "Grounding DINO-T  |  green = GT (sim mask)   red = prediction",
                               fill="#DDDDDD", font=BIG)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (8 + c*(W+8), 40 + 8 + r*(H+8)))
    out = "/workspace/gdino_sheet.png"
    sheet.save(out); print(f"\n모아보기 -> {out}  ({sheet.width}x{sheet.height})")

    print(f"\n{'task':20} {'uid':10} {'combined':>9} {'fixed':>8} {'separate':>9}")
    for task in sorted({r["task"] for r in rows}):
        for u in sorted({r["uid"] for r in rows if r["task"] == task}):
            g = lambda k: next((r["iou"] for r in rows if r["task"]==task and r["uid"]==u and r["mode"]==k), None)
            print(f"{task:20} {u:10} {g('combined'):9.3f} {g('fixed'):8.3f} {g('separate'):9.3f}")
    for k in ("combined", "fixed", "separate"):
        v = [r["iou"] for r in rows if r["mode"] == k]
        ms = sorted({(r["task"], r["ms"]) for r in rows if r["mode"] == k})
        print(f"\n{k:9} IoU>=0.5 {sum(1 for x in v if x>=0.5)}/{len(v)}  "
              f"평균 {np.mean(v):.3f}  지연중앙값 {np.median([m for _, m in ms]):.0f}ms")
    pathlib.Path("/workspace/gdino_modes.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
