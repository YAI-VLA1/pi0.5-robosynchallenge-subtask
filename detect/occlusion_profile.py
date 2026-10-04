#!/usr/bin/env python3
"""대본 에피소드를 한 번 돌리며 물체가 매 스텝 몇 픽셀 보이는지 기록한다.

왜
  "bbox 를 첫 프레임에만 줄 것인가, 매 스텝 줄 것인가"는 가림 정도에 달렸다.
  일반론 대신 우리 태스크에서 직접 잰다. 마스크가 픽셀 수를 정확히 준다.

출력: occlusion_<task>.json  — 스텝별 uid별 (보이는 픽셀, bbox)
"""
import json, os, sys, pathlib
import numpy as np

R = "/workspace/rsc_ws/RoboSynChallenge"
sys.path.insert(0, os.path.join(R, "scripts"))
os.chdir(R)
import eval_policy as EP

TASK = sys.argv[1] if len(sys.argv) > 1 else "items_handover"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else f"/workspace/occlusion_{TASK}.json")

gym_cfg = json.load(open(f"{R}/configs/{TASK}/random/gym_config.json"))
act_cfg = json.load(open(f"{R}/configs/{TASK}/action_config.json"))
cams = [s for s in gym_cfg.get("sensor", []) if s.get("sensor_type") == "Camera"]
for s in cams:
    s["enable_mask"] = True
CAMS = [s["uid"] for s in cams]

cfg = {"num_envs": 1, "device": "cpu", "headless": True, "renderer": "hybrid", "gpu_id": 0}
env, gc = EP.make_env_from_configs(cfg, gym_cfg, act_cfg)
obs, info = env.reset()
u, sim = env.unwrapped, env.unwrapped.sim

SKIP = ("cam_", "light", "_mat", "wall", "default_plane", "table", "CobotMagic", "contact", "background")
def scene_uids(d):
    out = []
    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("uid"), str): out.append(o["uid"])
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for x in o: walk(x)
    walk(d); return sorted(set(out))

targets = {}
for uid in scene_uids(gym_cfg):
    if any(k in uid for k in SKIP): continue
    try:
        targets[uid] = sim.get_asset(uid).get_user_ids().detach().cpu().numpy().ravel().tolist()
    except Exception: pass
# 그리퍼 링크도 따로 — 가림의 원인을 확인하기 위해
print("추적 대상:", targets, flush=True)

sensors = u.sensors
rows = []

def snap(step, seg):
    rec = {"step": step, "seg": seg}
    for cam in CAMS:
        sen = sensors[cam]; sen.update()
        m = sen._data_buffer["mask"]
        m = (m[0] if m.ndim == 3 else m).detach().cpu().numpy()
        for uid, ids in targets.items():
            sel = np.isin(m, ids); n = int(sel.sum())
            if n:
                ys, xs = np.where(sel)
                rec[f"{cam}|{uid}"] = [n, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            else:
                rec[f"{cam}|{uid}"] = [0, 0, 0, 0, 0]
    rows.append(rec)

step = 0
snap(step, -1)
for seg in range(60):                       # 대본 구간을 끝까지
    al = env.get_wrapper_attr("create_demo_action_list")(action_sentence=seg)
    if al is None or len(al) == 0:
        print(f"구간 {seg} 에서 종료 (총 {step} 스텝)", flush=True); break
    for a in al:
        obs, rew, term, trunc, info = env.step(a)
        step += 1
        snap(step, seg)
    print(f"  구간 {seg}: 누적 {step} 스텝", flush=True)

OUT.write_text(json.dumps({"task": TASK, "cams": CAMS, "targets": targets, "rows": rows}))
print(f"\nOCC-OK {len(rows)} 스텝 -> {OUT}")
