"""Core pieces of the RobotNav closed-loop eval suite.

Runs under the HABITAT env (robotnav-sim, py3.9). The policy itself runs in a
separate process under the lfm4vla env (see policy_server.py / PolicyClient).

Pieces: dataset paths, simulator builder (fixed nominal camera), waypoint
follower controller, nav metrics, and the rollout video recorder.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ paths --
_D = "/home/teams/research/robotics/datasets/robotnav_sources"
PATHS = {
    "mp3d": os.environ.get("ROBOTNAV_MP3D", f"{_D}/scene_datasets/mp3d"),
    "hm3d": os.environ.get("ROBOTNAV_HM3D", f"{_D}/scene_datasets/hm3d"),
    "vlnce_r2r": os.environ.get("ROBOTNAV_VLNCE_R2R", f"{_D}/vlnce_r2r/R2R_VLNCE_v1-3"),
    "vlnce_rxr": os.environ.get("ROBOTNAV_VLNCE_RXR", f"{_D}/vlnce_rxr/RxR_VLNCE_v0"),
    "objectnav_hm3d": os.environ.get("ROBOTNAV_OBJNAV_HM3D",
                                     f"{_D}/objectnav_hm3d_v2"),
    "objectnav_mp3d": os.environ.get("ROBOTNAV_OBJNAV_MP3D",
                                     f"{_D}/objectnav_mp3d_v1"),
    "lfm4vla_python": os.environ.get(
        "LFM4VLA_PYTHON",
        os.path.expanduser("~/miniconda3/envs/lfm4vla/bin/python")),
    "lfm4vla_root": os.environ.get("LFM4VLA_ROOT", os.path.expanduser("~/LFM4VLA")),
}


def mp3d_glb(scene: str) -> str:
    return f"{PATHS['mp3d']}/{scene}/{scene}.glb"


def hm3d_glb(scene_dir_name: str, split: str) -> str:
    d = Path(PATHS["hm3d"]) / split / scene_dir_name
    hashed = scene_dir_name.split("-")[-1]
    return str(d / f"{hashed}.basis.glb")


# ------------------------------------------------------------- simulator ---
def _egl_software_device_index() -> int:
    import ctypes
    try:
        E = ctypes.CDLL("libEGL.so.1")
        E.eglGetProcAddress.restype = ctypes.c_void_p
        qd_p = E.eglGetProcAddress(b"eglQueryDevicesEXT")
        qs_p = E.eglGetProcAddress(b"eglQueryDeviceStringEXT")
        if not (qd_p and qs_p):
            return 0
        qd = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_void_p),
                              ctypes.POINTER(ctypes.c_int))(qd_p)
        qs = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(qs_p)
        n = ctypes.c_int()
        devs = (ctypes.c_void_p * 256)()
        if not qd(256, devs, ctypes.byref(n)) or n.value == 0:
            return 0
        for i in range(n.value):
            if b"software" in (qs(devs[i], 0x3055) or b""):
                return i
    except Exception:
        pass
    return 0


def make_eval_sim(scene_glb: str, turn_deg: float, forward_m: float = 0.25,
                  height: float = 1.0, hfov: float = 105.0,
                  resolution=(400, 640)):
    """Front-camera sim with the nominal (median-of-training) camera config."""
    import habitat_sim
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene_glb)
    cfg.enable_physics = False
    cfg.gpu_device_id = -1
    if os.environ.get("ROBOTNAV_GPU_DEVICE_ID", "auto") == "auto":
        os.environ.setdefault("MAGNUM_DEVICE", str(_egl_software_device_index()))
    agent = habitat_sim.agent.AgentConfiguration()
    s = habitat_sim.CameraSensorSpec()
    s.uuid = "front"
    s.sensor_type = habitat_sim.SensorType.COLOR
    s.resolution = list(resolution)
    s.position = [0.0, height, 0.0]
    s.orientation = [0.0, 0.0, 0.0]
    s.hfov = hfov
    agent.sensor_specifications = [s]
    agent.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=forward_m)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_deg)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_deg)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
    return sim


def set_agent(sim, position, rotation_quat):
    import habitat_sim
    st = habitat_sim.AgentState()
    st.position = np.asarray(position, dtype=np.float32)
    st.rotation = rotation_quat
    sim.get_agent(0).set_state(st)


def geodesic(sim, a, b) -> float:
    import habitat_sim
    p = habitat_sim.ShortestPath()
    p.requested_start = np.asarray(a, dtype=np.float32)
    p.requested_end = np.asarray(b, dtype=np.float32)
    sim.pathfinder.find_path(p)
    return float(p.geodesic_distance)



def as_goals(goal) -> list:
    """Normalise a goal spec to a list of positions.

    VLN episodes have one goal; ObjectNav episodes have every instance of the
    target category and reaching ANY of them is success.
    """
    if goal is None:
        return []
    first = goal[0] if len(goal) else None
    return list(goal) if isinstance(first, (list, tuple, np.ndarray)) else [goal]


def geodesic_min(sim, a, goals):
    """Shortest geodesic from a to the nearest goal instance."""
    best = float("inf")
    for g in as_goals(goals):
        d = geodesic(sim, a, g)
        if math.isfinite(d) and d < best:
            best = d
    return best


def nearest_goal(sim, a, goals):
    """The goal instance with the shortest geodesic from a (for HUD/overlays)."""
    gs = as_goals(goals)
    if not gs:
        return None
    if len(gs) == 1:
        return gs[0]
    return min(gs, key=lambda g: (lambda d: d if math.isfinite(d) else 1e9)(
        geodesic(sim, a, g)))


def densify_path(points, step_m: float = 0.5):
    """Interpolate a sparse world-space path so overlays draw smoothly."""
    pts = [np.asarray(p, dtype=np.float32) for p in points]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(float(np.linalg.norm(b - a)) / step_m))
        for i in range(n):
            out.append(a + (b - a) * (i / n))
    out.append(pts[-1])
    return out


def path_to_ego(points, state):
    """World points -> robot ego frame [(fwd, left, up), ...]."""
    import quaternion as qt
    pos = np.asarray(state.position)
    qc = state.rotation.conjugate()
    out = []
    for p in points:
        lp = qt.rotate_vectors(qc, np.asarray(p) - pos)
        out.append((float(-lp[2]), float(-lp[0]), float(lp[1])))
    return out


def next_path_dir(sim, state, goal, min_m: float = 0.4):
    """Ego-frame (fwd, left) direction of the geodesic shortest path — the
    heading a perfect navigator would take NOW (unlike the straight-line
    goal bearing, this respects walls/stairs). None if no path found."""
    import habitat_sim
    import quaternion as qt
    p = habitat_sim.ShortestPath()
    p.requested_start = np.asarray(state.position, dtype=np.float32)
    p.requested_end = np.asarray(goal, dtype=np.float32)
    if not sim.pathfinder.find_path(p) or len(p.points) < 2:
        return None
    v = None
    for pt in p.points[1:]:
        v = np.asarray(pt) - np.asarray(state.position)
        if math.hypot(float(v[0]), float(v[2])) >= min_m:
            break
    lp = qt.rotate_vectors(state.rotation.conjugate(), v)
    return (float(-lp[2]), float(-lp[0]))       # habitat local: fwd=-z, left=-x


def advance_plan(wps, action: str, forward_m: float, turn_rad: float):
    """Re-express a predicted plan in the robot's NEW ego frame after one action.

    Waypoints are relative to the pose they were predicted from, so executing
    several of them open-loop requires transforming the remainder by the motion
    just made -- otherwise the plan goes stale and the controller steers at
    where the target USED to be.

    forward d : x -= d
    turn by t : rotate positions by -t (CCW positive), yaw -= t
    """
    w = np.array(wps, dtype=np.float32, copy=True)
    if action == "move_forward":
        w[:, 0] -= forward_m
        return w
    t = turn_rad if action == "turn_left" else -turn_rad
    c, s_ = math.cos(t), math.sin(t)
    x, y = w[:, 0].copy(), w[:, 1].copy()
    w[:, 0] = x * c + y * s_          # rotate by -t
    w[:, 1] = -x * s_ + y * c
    w[:, 2] -= t
    return w


def agent_yaw(sim) -> float:
    q = sim.get_agent(0).get_state().rotation
    # habitat yaw about +Y; forward is -Z at yaw 0
    return math.atan2(2.0 * (q.w * q.y + q.x * q.z),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# ------------------------------------------------------------ controller ---
class WaypointController:
    """Receding-horizon follower: robot-frame waypoints -> discrete actions.

    Waypoint convention (matches training): x forward (m), y left (m),
    yaw (rad, CCW). Predicts STOP when the final waypoint stays within
    stop_radius (the policy has no explicit stop token).

    stop_radius is 0.50 m: per-waypoint accuracy is ~0.25 m ADE, so the old
    0.24 m threshold demanded sub-noise precision and the policy essentially
    never fired it (every held-out episode ran to the step cap while passing
    within 5 m of the goal).
    """

    def __init__(self, turn_deg: float, lookahead_m: float = 0.30,
                 stop_radius: float = 0.50, min_steps_before_stop: int = 7,
                 stop_mode: str = "geometric", static_tol_xy: float = 0.06,
                 static_tol_yaw: float = 0.10, static_slots: int = 3):
        self.turn_rad = math.radians(turn_deg)
        self.lookahead = lookahead_m
        self.stop_radius = stop_radius
        self.min_steps_before_stop = min_steps_before_stop
        # "geometric": the controller decides, from the last waypoint's distance
        #   to the robot (the original proxy).
        # "plan_static": the MODEL decides. Terminal-hold training teaches the
        #   net to repeat the last real waypoint across the remaining slots, so
        #   a stop request is a plan that stops changing. Reading that directly
        #   removes the hand-tuned radius from the decision.
        self.stop_mode = str(stop_mode)
        self.static_tol_xy = float(static_tol_xy)
        self.static_tol_yaw = float(static_tol_yaw)
        self.static_slots = int(static_slots)

    def _plan_is_static(self, w) -> bool:
        """Is the model asking to hold position?

        Measured on the training labels, the three regimes are far apart:
            hold          dxy = 0         dyaw = 0        (exactly)
            turn-in-place dxy = 0         dyaw >= 0.2618  (15 deg floor)
            moving        dxy >= 0.1201
        so the default tolerances (0.06 m, 0.10 rad) sit in an empty gap with
        ~2x margin on both axes -- this is a decoder, not a tuning knob.
        """
        k = min(self.static_slots, w.shape[0] - 1)
        if k < 1:
            return False
        tail = w[-(k + 1):]
        d = np.diff(tail, axis=0)
        dxy = np.linalg.norm(d[:, :2], axis=1)
        dyaw = np.abs(d[:, 2])
        return bool((dxy < self.static_tol_xy).all()
                    and (dyaw < self.static_tol_yaw).all())

    def act(self, waypoints: np.ndarray, fresh: bool = True,
            step: int = 1 << 30) -> str:
        """fresh=False when following a cached plan mid-cycle.

        `step` is the episode step index; the stop test is suppressed for the
        first min_steps_before_stop steps.

        The stop test is only meaningful on a NEWLY predicted plan ("from what
        I see now, I predict no motion"). While executing a cached plan the
        remaining waypoints shrink toward the robot by construction -- each
        move_forward subtracts 0.25 m -- so an 8-waypoint (~2 m) plan drops
        inside stop_radius after ~5 steps and would fire a false "arrived".
        """
        w = np.asarray(waypoints, dtype=np.float32)
        trans = float(np.linalg.norm(w[-1, :2]))
        yaw8 = float(w[-1, 2])
        # Training data encodes turn-in-place (e.g. episode-start turnarounds,
        # 18% of t=0 samples) as (0, 0, yaw) plans: position channels alone
        # cannot distinguish "arrived" from "rotate first". Stop only when the
        # plan is at rest in BOTH position and heading; execute the predicted
        # yaw when the plan is rotation-dominant.
        # Suppress the stop test at the very start of an episode. At
        # replan_every=1 the test runs against a fresh plan on step 0, so a
        # single under-confident prediction inside stop_radius ends the episode
        # before the robot moves at all (path_len 0.0, steps 1). Real arrivals
        # cannot occur in the first few steps from a valid start pose.
        if fresh and step >= self.min_steps_before_stop:
            if self.stop_mode == "plan_static":
                # The model owns the decision: it stops when it emits a plan
                # that no longer moves. No distance threshold is consulted.
                if self._plan_is_static(w):
                    return "stop"
            elif self.stop_mode == "both":
                if (self._plan_is_static(w) and trans < self.stop_radius
                        and abs(yaw8) < math.radians(12.0)):
                    return "stop"
            elif (trans < self.stop_radius
                  and abs(yaw8) < math.radians(12.0)):
                return "stop"
        if trans < self.lookahead and abs(yaw8) >= self.turn_rad / 2.0:
            return "turn_left" if yaw8 > 0 else "turn_right"
        tgt = None
        for i in range(w.shape[0]):
            if float(np.linalg.norm(w[i, :2])) >= self.lookahead:
                tgt = w[i, :2]
                break
        if tgt is None:
            tgt = w[-1, :2]
        heading_err = math.atan2(float(tgt[1]), float(tgt[0]))  # left-positive
        if abs(heading_err) > self.turn_rad / 2.0 + 1e-6:
            return "turn_left" if heading_err > 0 else "turn_right"
        return "move_forward"


# --------------------------------------------------------------- metrics ---
def episode_metrics(sim, goal_pos, path_positions, start_geo: float,
                    success_dist: float):
    """NE / SR / OS / SPL from a rollout's position trace.

    ``goal_pos`` may be one position or many. ObjectNav episodes list every
    instance of the target category and reaching any one of them counts, so
    distances are taken to the NEAREST instance rather than the first.
    """
    goals = as_goals(goal_pos)
    ne = geodesic_min(sim, path_positions[-1], goals)
    if not math.isfinite(ne):
        ne = min(float(np.linalg.norm(np.asarray(path_positions[-1]) - np.asarray(g)))
                 for g in goals)
    sr = float(ne <= success_dist)
    dists = []
    for p in path_positions[:: max(1, len(path_positions) // 50)]:
        d = geodesic_min(sim, p, goals)
        if math.isfinite(d):
            dists.append(d)
    os_ = float(min(dists) <= success_dist) if dists else 0.0
    length = float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a))
                       for a, b in zip(path_positions[:-1], path_positions[1:])))
    spl = sr * start_geo / max(start_geo, length, 1e-6) if math.isfinite(start_geo) else 0.0
    return {"ne_m": round(ne, 3), "sr": sr, "os": os_, "spl": round(spl, 4),
            "path_len_m": round(length, 2)}


# ----------------------------------------------------------------- video ---
class RolloutRecorder:
    def __init__(self, out_path: str, fps: int = 5):
        self.frames = []
        self.out = out_path
        self.fps = fps

    def add(self, rgb: np.ndarray, hud_lines, overlay=None):
        from PIL import Image, ImageDraw
        im = Image.fromarray(rgb[..., :3])
        d = ImageDraw.Draw(im)
        if overlay:
            self._draw_overlay(d, im.size, overlay)
        y = 4
        for line in hud_lines:
            d.rectangle([2, y - 2, 638, y + 12], fill=(0, 0, 0))
            d.text((6, y), line[:110], fill=(255, 255, 80))
            y += 15
        self.frames.append(np.asarray(im))

    @staticmethod
    def _draw_overlay(d, size, ov):
        """Waypoint diagnostics: floor-projected plan + bird's-eye inset.

        Ego convention (training): x forward, y left, yaw CCW — camera level
        at cam_height, so a waypoint (X, Y) sits on the floor at camera coords
        (right=-Y, down=cam_height, depth=X).
        """
        W, H = size
        wps = np.asarray(ov["wps"], dtype=np.float32)
        K = wps.shape[0]

        def col(i):  # early plan = green -> final waypoint = red
            f = i / max(1, K - 1)
            return (int(60 + 195 * f), int(230 - 180 * f), 60)

        # -- camera projection of the plan onto the floor ---------------------
        fx = (W / 2.0) / math.tan(math.radians(ov.get("hfov_deg", 105.0)) / 2)
        ch = ov.get("cam_height", 1.0)
        gt = ov.get("gt_path")                   # [(fwd, left, up), ...]
        if gt:
            prev = None
            for X, Y, Z in gt:                   # painted on the actual route
                if X < 0.3 or X > 12.0:          # behind / too far to draw
                    prev = None
                    continue
                q = (W / 2.0 + fx * (-Y) / X, H / 2.0 + fx * (ch - Z) / X)
                if prev is not None:
                    d.line([prev, q], fill=(255, 0, 255), width=3)
                prev = q
        pts = []
        for i in range(K):
            X, Y = float(wps[i, 0]), float(wps[i, 1])
            if X < 0.15:                       # behind / too close to project
                pts.append(None)
                continue
            u = W / 2.0 + fx * (-Y) / X
            v = H / 2.0 + fx * ch / X
            pts.append((u, v))
        prev = None
        for i, p in enumerate(pts):
            if p is None:
                prev = None
                continue
            if prev is not None:
                d.line([prev, p], fill=col(i), width=2)
            r = 5 if i == K - 1 else 3
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                      outline=col(i), width=2)
            prev = p

        # -- bird's-eye inset (bottom-right): plan vs goal vs controller -----
        S = 148
        x0, y0 = W - S - 6, H - S - 6
        cx, cy = x0 + S / 2, y0 + S / 2
        d.rectangle([x0, y0, x0 + S, y0 + S], fill=(0, 0, 0),
                    outline=(200, 200, 200))
        rmax = max(2.5, float(np.abs(wps[:, :2]).max()) * 1.15)
        ppm = (S / 2 - 8) / rmax

        def bev(X, Y):                          # ego fwd = up, ego left = left
            return (cx - Y * ppm, cy - X * ppm)

        for rad, c in ((ov.get("stop_radius", 0.24), (120, 120, 120)),
                       (ov.get("lookahead", 0.30), (80, 80, 160))):
            rp = rad * ppm
            d.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], outline=c)
        if gt:                                   # GT route, clipped to inset
            prev = None
            lim = rmax * 0.98
            for X, Y, _ in gt:
                if abs(X) > lim or abs(Y) > lim:
                    prev = None
                    continue
                q = bev(X, Y)
                if prev is not None:
                    d.line([prev, q], fill=(255, 0, 255), width=2)
                prev = q
        d.polygon([bev(0.12, 0), bev(-0.08, 0.07), bev(-0.08, -0.07)],
                  fill=(255, 255, 255))         # robot, nose up
        prev = bev(0, 0)
        for i in range(K):
            p = bev(wps[i, 0], wps[i, 1])
            d.line([prev, p], fill=col(i), width=2)
            d.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=col(i))
            yaw = float(wps[i, 2])              # heading tick, CCW from +x
            d.line([p, (p[0] - 7 * math.sin(yaw), p[1] - 7 * math.cos(yaw))],
                   fill=col(i))
            prev = p
        ge = ov.get("goal_ego")
        if ge is not None:                       # distance label only; the
            n = math.hypot(float(ge[0]), float(ge[1]))   # magenta LINE is GT
            d.text((x0 + 4, y0 + 2), f"goal {n:.1f}m", fill=(255, 0, 255))
        pe = ov.get("path_ego")                  # geodesic next-step heading
        if pe is not None:
            px, py = float(pe[0]), float(pe[1])
            s = rmax * 0.6 / max(math.hypot(px, py), 1e-6)
            d.line([bev(0, 0), bev(px * s, py * s)], fill=(0, 220, 255),
                   width=2)
            d.text((x0 + 4, y0 + 14), "path", fill=(0, 220, 255))
        d.text((x0 + 4, y0 + S - 14), ov.get("action", ""),
               fill=(255, 255, 80))

    def save(self):
        Path(self.out).parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.mimwrite(self.out, self.frames, fps=self.fps,
                             quality=7, macro_block_size=1)
            return self.out
        except Exception:
            d = Path(self.out).with_suffix("")
            d.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            for i, f in enumerate(self.frames):
                Image.fromarray(f).save(d / f"{i:04d}.jpg", quality=85)
            return str(d) + "/ (PNG frames; install imageio-ffmpeg for mp4)"


# ---------------------------------------------------------- policy client ---
class PolicyClient:
    """Spawns policy_server.py under the lfm4vla env; JSON-lines over stdio."""

    def __init__(self, config: str, ckpt: str, device: str = "cuda"):
        server = Path(__file__).parent / "policy_server.py"
        self.proc = subprocess.Popen(
            [PATHS["lfm4vla_python"], str(server), "--config", config,
             "--ckpt", ckpt, "--device", device],
            cwd=PATHS["lfm4vla_root"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        ready = ""
        for _ in range(1000):        # tolerate startup chatter before READY
            line = self.proc.stdout.readline()
            if not line:
                break
            ready = line.strip()
            if ready.startswith("READY"):
                break
        if not ready.startswith("READY"):
            raise RuntimeError(f"policy server failed to start: {ready!r}")

    def predict(self, instruction: str, frames, family: str,
                frame_ids=None) -> np.ndarray:
        payload = {"instruction": instruction, "family": family, "frames": [],
                   "frame_ids": list(frame_ids) if frame_ids else None}
        for f in frames:
            buf = io.BytesIO()
            f.save(buf, format="JPEG", quality=92)
            payload["frames"].append(base64.b64encode(buf.getvalue()).decode())
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:                  # skip any non-protocol chatter
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("policy server exited mid-episode")
            line = line.strip()
            if line.startswith("{"):
                resp = json.loads(line)
                break
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return np.asarray(resp["waypoints"], dtype=np.float32)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass
