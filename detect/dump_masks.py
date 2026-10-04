#!/usr/bin/env python3
"""sim 에서 instance 마스크를 켜고 한 에피소드를 띄워 GT bbox 를 뽑는다.

왜
  sim 데이터는 렌더러가 instance ID 를 직접 내준다. 탐지기도, 라벨링도, 오탐도 없다.
  real 데이터에는 이게 영원히 없으므로, 여기서 얻은 값이 탐지기 채점의 정답지가 된다.

확인하려는 것
  1. 이 버전 EmbodiChain 에서 enable_mask 가 실제로 도는가
  2. instance ID 가 물체 uid 로 되짚어지는가 (asset.get_user_ids())
  3. duck / milk / cube 가 이름대로 생겼는가  ← 눈으로 봐야 한다
  4. 카메라 파라미터가 런타임에 읽히는가 (기존 데이터 pose 투영 경로의 가부)
"""
import json, os, sys, pathlib
import numpy as np

R = "/workspace/rsc_ws/RoboSynChallenge"
sys.path.insert(0, os.path.join(R, "scripts"))
os.chdir(R)
import eval_policy as EP          # 경로·매니저 모듈 등록을 여기서 한다 (import 순서 중요)

TASK = sys.argv[1] if len(sys.argv) > 1 else "items_handover"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else f"/workspace/masks_{TASK}")
OUT.mkdir(parents=True, exist_ok=True)

gym_cfg = json.load(open(f"{R}/configs/{TASK}/random/gym_config.json"))
act_cfg = json.load(open(f"{R}/configs/{TASK}/action_config.json"))

cams = [s for s in gym_cfg.get("sensor", []) if s.get("sensor_type") == "Camera"]
for s in cams:
    s["enable_mask"] = True
print(f"카메라 {len(cams)}대에 enable_mask 켬:", [s["uid"] for s in cams], flush=True)

cfg = {"num_envs": 1, "device": "cuda:0", "headless": True, "renderer": "hybrid", "gpu_id": 0}
env, gc = EP.make_env_from_configs(cfg, gym_cfg, act_cfg)
obs, info = env.reset()
print("reset OK", flush=True)

u = env.unwrapped
sim = u.sim

# ── 물체 uid -> mask 정수값 ──────────────────────────────────────────────
SKIP = ("cam_", "light", "_mat", "wall", "default_plane")

def scene_uids(d):
    out = []
    def walk(o):
        if isinstance(o, dict):
            if "uid" in o and isinstance(o["uid"], str): out.append(o["uid"])
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for x in o: walk(x)
    walk(d); return sorted(set(out))

cand = [x for x in scene_uids(gym_cfg) if not any(k in x for k in SKIP)]
print("\n후보 uid:", cand, flush=True)

id_map = {}
for uid in cand:
    try:
        a = sim.get_asset(uid)
    except Exception as e:
        print(f"  {uid:28} asset 없음 ({type(e).__name__})"); continue
    try:
        ids = a.get_user_ids()
        ids = ids.detach().cpu().numpy().ravel().tolist()
        id_map[uid] = ids
        print(f"  {uid:28} user_ids={ids}")
    except Exception as e:
        print(f"  {uid:28} user_ids 실패 ({type(e).__name__}: {e})")

# ── 마스크 가져오기 ──────────────────────────────────────────────────────
print("\n=== 센서에서 마스크 꺼내기 ===", flush=True)
sensors = getattr(u, "sensors", None) or getattr(u.scene, "sensors", None) or {}
print("sensor 핸들:", list(sensors) if hasattr(sensors, "__iter__") else type(sensors))

import imageio.v3 as iio
for name in (s["uid"] for s in cams):
    try:
        sen = sensors[name] if not callable(sensors) else sensors(name)
    except Exception:
        try: sen = sim.get_sensor(name)
        except Exception as e:
            print(f"{name}: 센서 접근 실패 {e}"); continue
    sen.update()
    buf = getattr(sen, "_data_buffer", {})
    print(f"\n[{name}] 버퍼 키: {list(buf)}")
    if "mask" not in buf: print("  mask 없음"); continue
    m = buf["mask"][0].detach().cpu().numpy()
    print(f"  mask shape={m.shape} dtype={m.dtype} 고유값 {len(np.unique(m))}개: {np.unique(m)[:25]}")
    np.save(OUT / f"{name}_mask.npy", m)
    if "color" in buf:
        c = buf["color"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
        iio.imwrite(OUT / f"{name}_rgb.png", c)
    # uid 별 bbox
    print(f"  {'uid':24} {'픽셀':>8}  bbox(x0,y0,x1,y1)")
    for uid, ids in id_map.items():
        sel = np.isin(m, ids)
        n = int(sel.sum())
        if n == 0: print(f"  {uid:24} {0:>8}  (안 보임)"); continue
        ys, xs = np.where(sel)
        print(f"  {uid:24} {n:>8}  ({xs.min()},{ys.min()},{xs.max()},{ys.max()})")

# ── 카메라 파라미터가 런타임에 읽히는가 ────────────────────────────────
print("\n=== 카메라 파라미터 ===", flush=True)
for name in (s["uid"] for s in cams):
    try:
        sen = sensors[name]
        for attr in ("intrinsics", "extrinsics", "get_intrinsics", "get_extrinsics", "cfg"):
            v = getattr(sen, attr, None)
            if v is None: continue
            if callable(v): v = v()
            print(f"  [{name}] {attr}: {str(v)[:220]}")
    except Exception as e:
        print(f"  {name}: {e}")

print(f"\nDUMP-OK -> {OUT}")
