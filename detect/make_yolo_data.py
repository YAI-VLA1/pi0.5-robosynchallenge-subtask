#!/usr/bin/env python3
"""대본 에피소드를 돌리며 YOLO 학습 데이터(깨끗한 RGB + 박스 라벨)를 만든다.

라벨은 렌더러의 instance 마스크에서 나온다 — 사람 라벨링도, 탐지기도 없다.

distractor 처리
  distractor 는 리셋마다 instance id 가 바뀌어 uid 로 못 찾는다
  (get_asset('distractor_0').get_user_ids() 가 주는 103/105 는 안 찍히고
   실제로는 129 같은 값이 찍힌다). 그래서 **알려진 에셋이 아닌 id 는 전부
  distractor** 로 본다. 이게 'fork 를 pen 으로 보는' 오탐을 고치는 핵심 라벨이다.

클래스: 0=pen 1=holder 2=distractor
"""
import json, os, sys, pathlib
import numpy as np
from PIL import Image

R = "/workspace/rsc_ws/RoboSynChallenge"
sys.path.insert(0, os.path.join(R, "scripts")); os.chdir(R)
import eval_policy as EP

TASK = "items_handover"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/yolo_ds")
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 15
EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 10     # sim step 간격
VAL_EVERY = 5                                             # 5 에피소드마다 val
CLS = {"pen": 0, "holder": 1}
MIN_PX = 40                                               # 너무 작으면 라벨 노이즈

g = json.load(open(f"{R}/configs/{TASK}/random/gym_config.json"))
a = json.load(open(f"{R}/configs/{TASK}/action_config.json"))
cams = [s for s in g.get("sensor", []) if s.get("sensor_type") == "Camera"]
for s in cams: s["enable_mask"] = True
CAMS = [s["uid"] for s in cams]

env, _ = EP.make_env_from_configs(
    {"num_envs":1,"device":"cpu","headless":True,"renderer":"hybrid","gpu_id":0}, g, a)
u = env.unwrapped; sim = u.sim

def all_uids(d):
    out=[]
    def w(o):
        if isinstance(o,dict):
            if isinstance(o.get("uid"),str): out.append(o["uid"])
            for v in o.values(): w(v)
        elif isinstance(o,list):
            for x in o: w(x)
    w(d); return sorted(set(out))

for sp in ("train", "val"):
    (OUT/"images"/sp).mkdir(parents=True, exist_ok=True)
    (OUT/"labels"/sp).mkdir(parents=True, exist_ok=True)

n_img = {"train": 0, "val": 0}
n_box = {0: 0, 1: 0, 2: 0}

for ep in range(N_EP):
    env.reset()
    split = "val" if ep % VAL_EVERY == 4 else "train"
    # 이번 에피소드의 id 매핑 (리셋마다 다시 읽는다 — distractor id 가 바뀐다)
    known, want = {}, {}
    for x in all_uids(g):
        try:
            ids = sim.get_asset(x).get_user_ids().detach().cpu().numpy().ravel().tolist()
        except Exception:
            continue
        for i in ids:
            known[int(i)] = x
        if x in CLS:
            want[x] = [int(i) for i in ids]

    step = 0
    for seg in range(60):
        al = env.get_wrapper_attr("create_demo_action_list")(action_sentence=seg)
        if al is None or len(al) == 0: break
        for act in al:
            env.step(act); step += 1
            if step % EVERY: continue
            for cam in CAMS:
                sen = u.sensors[cam]; sen.update()
                m = sen._data_buffer["mask"]
                m = (m[0] if m.ndim == 3 else m).detach().cpu().numpy()
                H, W = m.shape
                lines = []
                def put(c, sel):
                    if sel.sum() < MIN_PX: return
                    ys, xs = np.where(sel)
                    x0,y0,x1,y1 = xs.min(), ys.min(), xs.max(), ys.max()
                    lines.append(f"{c} {((x0+x1)/2)/W:.6f} {((y0+y1)/2)/H:.6f} "
                                 f"{(x1-x0+1)/W:.6f} {(y1-y0+1)/H:.6f}")
                    n_box[c] += 1
                for uid, c in CLS.items():
                    if uid in want: put(c, np.isin(m, want[uid]))
                # 알려지지 않은 id = distractor (id 가 매번 바뀌므로 이렇게만 잡힌다)
                for i in np.unique(m):
                    if int(i) not in known:
                        put(2, m == i)
                if not lines: continue
                name = f"ep{ep:03d}_{cam}_{step:04d}"
                c3 = sen._data_buffer["color"]
                c3 = (c3[0] if c3.ndim == 4 else c3).detach().cpu().numpy()[..., :3]
                Image.fromarray(c3.astype("uint8")).save(OUT/"images"/split/f"{name}.png")
                (OUT/"labels"/split/f"{name}.txt").write_text("\n".join(lines))
                n_img[split] += 1
    print(f"  ep{ep:03d} [{split}] 누적 train {n_img['train']} val {n_img['val']} "
          f"| pen {n_box[0]} holder {n_box[1]} distractor {n_box[2]}", flush=True)

(OUT/"data.yaml").write_text(
    f"path: {OUT}\ntrain: images/train\nval: images/val\n"
    f"names:\n  0: pen\n  1: holder\n  2: distractor\n")
print(f"\nYOLO-DATA-DONE  train {n_img['train']} val {n_img['val']}  "
      f"박스 pen {n_box[0]} holder {n_box[1]} distractor {n_box[2]}")
print(f"-> {OUT}/data.yaml")
