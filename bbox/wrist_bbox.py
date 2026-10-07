#!/usr/bin/env python3
"""손목 카메라(cam_right_wrist)에 펜 bbox 를 투영한다.

왜 cam_high 와 다른가
  cam_high 는 고정이라 gym_config 의 eye/target/up 으로 끝난다.
  손목 카메라는 **right_link6 에 붙어** 있어서, 매 프레임 qpos 로
  순기구학을 풀어 그 링크의 월드 pose 를 구해야 한다.

  카메라 = link6 pose · (pos, quat) 오프셋
  gym_config: parent right_link6, pos [-0.12, 0.01, 0.03],
              quat [0.1949, 0.6797, -0.6797, -0.1949]

왜 필요한가 (실측)
  cam_high 는 펜을 들어올린 뒤 화면 밖으로 놓친다 — 왼팔 운반 구간 29.7%,
  홀더 꽂기 21.1%. 그 구간에 손목 카메라는 펜을 보고 있을 가능성이 크다.
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import numpy as np

sys.path.insert(0, "/root/rsc_recover")
URDF_CANDIDATES = [
    "/workspace/.cache/embodichain_data/assembled/"
    "CobotMagicWithGripperV100_CobotMagicWithGripperV100/"
    "CobotMagicWithGripperV100_CobotMagicWithGripperV100.urdf",
    # 시뮬레이터를 안 깔고 자산만 받았을 때. CobotMagicArmV3.zip 은 extract/ 바로
    # 아래로 풀린다 (bbox_run/fetch_urdf.sh). 조립본과 같은 V100 이다.
    "/workspace/.cache/embodichain_data/extract/CobotMagicWithGripperV100.urdf",
    "/workspace/.cache/embodichain_data/extract/CobotMagicArm/"
    "CobotMagicWithGripperV70NoMaterial.urdf",
]
CFG = "/workspace/rsc_ws/RoboSynChallenge/configs/items_handover/random/gym_config.json"
W, H = 640, 480


# 탐색으로 확정한 규약 (2026-10-06).
#   gym_config 의 quat 은 **wxyz** 순이고, 카메라 축은 OpenGL(-z 전방) 이라
#   OpenCV(+z 전방) 로 바꾸려면 y·z 를 뒤집어야 한다.
#   접근~집기 40/40 프레임이 화면 안에 들어오고, 실제 손목 영상과 정확히 일치한다
#   (/workspace/wrist_vis.png 로 눈으로 확인).
QUAT_ORDER = "wxyz"
AXIS_FIX = np.diag([1.0, -1.0, -1.0])


def quat_to_R(q, order=QUAT_ORDER):
    """gym_config 의 quat -> 회전 행렬."""
    if order == "xyzw":
        x, y, z, w = q
    else:
        w, x, y, z = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]], float)


def load_robot():
    import pinocchio as pin
    for p in URDF_CANDIDATES:
        if pathlib.Path(p).exists():
            m = pin.buildModelFromUrdf(p)
            return pin, m, m.createData(), p
    raise SystemExit("★ CobotMagic URDF 를 못 찾았다")



# ── 순기구학 + 투영 ────────────────────────────────────────────────────────
# 로봇 베이스가 월드에서 init_pos 만큼 올라가 있다 (gym_config: [0,0,0.835]).
# URDF 는 베이스 기준이므로 그 평행이동을 더해야 월드 좌표가 된다.
def robot_base():
    g = json.load(open(CFG))
    r = g.get("robot")
    if isinstance(r, dict):                 # 단일 로봇이면 dict 다 (리스트가 아니다)
        r = [r]
    r = [o for o in r if isinstance(o, dict) and o.get("uid") == "CobotMagic"][0]
    return np.asarray(r.get("init_pos", [0, 0, 0]), float)


# 오른팔이 베이스에 붙는 자리. EmbodiChain 의 CobotMagicCfg.right_arm_xpos 는
# 순수 평행이동 [0.233, -0.300, 0] 이다 (회전 없음).
RIGHT_ARM_XPOS = np.array([0.233, -0.300, 0.0])


def qpos_to_urdf(q14, nq=16):
    """관측 state(14축) -> URDF 관절 벡터.

    관측은 [LEFT_JOINT1..7, RIGHT_JOINT1..7] 이고 7번째가 그리퍼다.
    URDF 의 손가락은 둘이라 그리퍼 값을 양쪽에 같이 넣는다. 카메라는 link6 에
    붙어 있어 손가락 값과는 무관하다.

    nq=16  조립본(left_* + right_*). 원래 쓰던 모델이다.
    nq=8   단일 팔 모델. 시뮬레이터를 안 깔면 조립본이 없어서 이쪽을 쓴다 —
           오른팔만 넣고, 베이스 오프셋은 호출자가 더한다.
    """
    q14 = np.asarray(q14, float)
    if nq == 8:
        q = np.zeros(8, float)
        q[0:6] = q14[7:13]; q[6] = q14[13]; q[7] = q14[13]
        return q
    q = np.zeros(16, float)
    q[0:6] = q14[0:6]; q[6] = q14[6]; q[7] = q14[6]
    q[8:14] = q14[7:13]; q[14] = q14[13]; q[15] = q14[13]
    return q


class WristCam:
    def __init__(self):
        self.pin, self.model, self.data, _ = load_robot()
        # 조립본이면 right_link6, 단일 팔 모델이면 link6 다. 단일 팔은 자기 베이스
        # 기준이라 오른팔 장착 위치를 더해야 조립본과 같은 좌표가 된다.
        self.nq = self.model.nq
        self.fid = self.model.getFrameId("right_link6" if self.nq == 16 else "link6")
        self.base = robot_base() + (RIGHT_ARM_XPOS if self.nq == 8 else 0.0)
        g = json.load(open(CFG))
        c = [s for s in g["sensor"] if s["uid"] == "cam_right_wrist"][0]
        fx, fy, cx, cy = c["intrinsics"]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        e = c["extrinsics"]
        self.off_p = np.asarray(e["pos"], float)
        self.off_R = quat_to_R(e["quat"]) @ AXIS_FIX

    def cam_pose(self, q14):
        q = qpos_to_urdf(q14, self.nq)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        M = self.data.oMf[self.fid]
        Rl, tl = np.asarray(M.rotation), np.asarray(M.translation) + self.base
        Rc = Rl @ self.off_R
        tc = tl + Rl @ self.off_p
        return Rc, tc

    def project(self, q14, pts_w):
        """월드 점 -> 손목 카메라 픽셀. (uv, z) 를 돌려준다."""
        Rc, tc = self.cam_pose(q14)
        p = (Rc.T @ (np.asarray(pts_w) - tc).T).T          # 월드 -> 카메라
        z = np.maximum(p[:, 2], 1e-6)
        uv = (self.K @ (p / z[:, None]).T).T[:, :2]
        return uv, p[:, 2]



# ── 라벨 추출 ──────────────────────────────────────────────────────────────
def extract(out_dir, n_ep=1000):
    """학습 데이터 전체에 손목 bbox 4토큰 라벨을 만든다.

    출력은 cam_high 와 같은 형식이다: [ymin, xmin, ymax, xmax, visible] int16.
      visible 1 = 실제 박스, 2 = "박스 없음"(화면 밖·뒤쪽), 0 = 쓰지 않음
    """
    import pandas as pd
    from obb_labels import pen_corners
    from make_bbox_labels import quant, NO_BOX_TOK, VIS_BOX, VIS_NONE

    DS = pathlib.Path("/workspace/rsc_ws/RoboSynChallenge/policy/pi05/training_data"
                      "/RoboSynChallenge/cobotmagic_Sim_items_handover")
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cam = WristCam(); C = pen_corners()
    nvis = ntot = 0
    for ep in range(n_ep):
        pq = DS / f"data/chunk-000/episode_{ep:06d}.parquet"
        if not pq.exists():
            break
        df = pd.read_parquet(pq, columns=["observation.state", "pen_pose"])
        st = np.stack(df["observation.state"].to_numpy()).astype(float)
        P4 = np.stack([np.stack([np.asarray(r, float) for r in m]) for m in df["pen_pose"]])
        t = np.zeros((len(st), 5), np.int16)
        for i in range(len(st)):
            ntot += 1
            M = P4[i]
            pw = (M[:3, :3] @ C.T).T + M[:3, 3]
            uv, z = cam.project(st[i], pw)
            if (z <= 0).any():                       # 카메라 뒤
                t[i] = NO_BOX_TOK + (VIS_NONE,); continue
            x0, y0 = uv[:, 0].min()/W, uv[:, 1].min()/H
            x1, y1 = uv[:, 0].max()/W, uv[:, 1].max()/H
            cx0, cy0 = max(x0, 0.0), max(y0, 0.0)
            cx1, cy1 = min(x1, 1.0), min(y1, 1.0)
            if cx1 <= cx0 or cy1 <= cy0:             # 완전히 화면 밖
                t[i] = NO_BOX_TOK + (VIS_NONE,); continue
            t[i] = (quant(cy0), quant(cx0), quant(cy1), quant(cx1), VIS_BOX)
            nvis += 1
        np.save(out / f"ep{ep:04d}.npy", t)
        if (ep + 1) % 100 == 0:
            print(f"  {ep+1} 에피소드", flush=True)
    print(f"프레임 {ntot:,} · 박스 있는 프레임 {nvis:,} ({nvis/max(ntot,1):.1%})")
    print(f"  '박스 없음' {ntot-nvis:,} ({1-nvis/max(ntot,1):.1%})")
    print(f"-> {out}")
    print("WRIST-LABELS-DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="조인트·링크 이름만 찍는다")
    ap.add_argument("--extract", default="", help="라벨을 뽑을 디렉터리")
    ap.add_argument("--n-ep", type=int, default=1000)
    a = ap.parse_args()
    pin, model, data, path = load_robot()
    print(f"URDF {path}")
    print(f"  nq={model.nq} nv={model.nv} · 조인트 {model.njoints} · 프레임 {model.nframes}")
    names = [model.names[i] for i in range(model.njoints)]
    print(f"  조인트: {names[:20]}")
    fr = [model.frames[i].name for i in range(model.nframes)]
    cand = [n for n in fr if "link6" in n.lower() or "link_6" in n.lower()]
    print(f"  link6 후보: {cand}")
    if a.probe:
        print(f"  전체 프레임 (앞 40): {fr[:40]}")
        return
    g = json.load(open(CFG))
    c = [s for s in g["sensor"] if s["uid"] == "cam_right_wrist"][0]
    print(f"  cam extrinsics {c['extrinsics']}")
    if a.extract:
        extract(a.extract, a.n_ep); return
    print("WRIST-PROBE-DONE")


if __name__ == "__main__":
    main()
