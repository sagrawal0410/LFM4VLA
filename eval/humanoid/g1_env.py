"""Unitree G1 + Dex3 MuJoCo env for HE closed-loop eval.

Loads Menagerie ``g1_with_hands.xml``, adds a tabletop + task objects, PD-holds
legs/waist at the stand keyframe, and drives the 28 policy joints from absolute
targets. Soft-welds the pelvis so the floating base does not tip immediately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval.humanoid.g1_joint_map import (
    ACTION_DIM,
    LEG_WAIST_JOINTS,
    POLICY_JOINTS,
    default_robot_xml,
    menagerie_unitree_g1_dir,
)


@dataclass
class TaskSpec:
    name: str
    suite: str  # "in" | "ood"
    instruction: str
    max_steps: int = 400
    # Named free objects placed at reset: name -> (pos xyz, quat wxyz, size/extra)
    objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    success_type: str = "object_in_bin"
    success_params: Dict[str, Any] = field(default_factory=dict)


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")


def _object_body_xml(name: str, spec: Dict[str, Any]) -> str:
    """Build a free-joint body snippet for a task object."""
    pos = spec.get("pos", [0.45, 0.0, 0.85])
    quat = spec.get("quat", [1, 0, 0, 0])
    rgba = spec.get("rgba", [0.8, 0.2, 0.2, 1])
    mass = float(spec.get("mass", 0.1))
    kind = spec.get("geom", "box")
    friction = spec.get("friction", "1 0.005 0.0001")
    pos_s = " ".join(f"{x:.4f}" for x in pos)
    quat_s = " ".join(f"{x:.4f}" for x in quat)
    rgba_s = " ".join(str(x) for x in rgba)
    n = _xml_escape(name)

    if kind == "box":
        size = spec.get("size", [0.03, 0.03, 0.03])
        size_s = " ".join(f"{x:.4f}" for x in size)
        geom = f'<geom name="{n}_geom" type="box" size="{size_s}" rgba="{rgba_s}" friction="{friction}" mass="{mass}"/>'
    elif kind == "cylinder":
        size = spec.get("size", [0.03, 0.06])  # radius, half-height
        size_s = " ".join(f"{x:.4f}" for x in size)
        geom = f'<geom name="{n}_geom" type="cylinder" size="{size_s}" rgba="{rgba_s}" friction="{friction}" mass="{mass}"/>'
    elif kind == "sphere":
        size = spec.get("size", [0.03])
        geom = f'<geom name="{n}_geom" type="sphere" size="{size[0]:.4f}" rgba="{rgba_s}" friction="{friction}" mass="{mass}"/>'
    elif kind == "capsule":
        size = spec.get("size", [0.02, 0.05])
        size_s = " ".join(f"{x:.4f}" for x in size)
        geom = f'<geom name="{n}_geom" type="capsule" size="{size_s}" rgba="{rgba_s}" friction="{friction}" mass="{mass}"/>'
    else:
        raise ValueError(f"unknown geom kind {kind}")

    # Optional hinge child (drawer / laptop lid / button travel).
    extra = ""
    if "hinge" in spec:
        h = spec["hinge"]
        axis = " ".join(str(x) for x in h.get("axis", [0, 1, 0]))
        jrange = " ".join(str(x) for x in h.get("range", [0, 1.2]))
        child_pos = " ".join(f"{x:.4f}" for x in h.get("pos", [0, 0, 0.02]))
        child_size = " ".join(f"{x:.4f}" for x in h.get("size", [0.12, 0.08, 0.01]))
        extra = f"""
      <body name="{n}_lid" pos="{child_pos}">
        <joint name="{n}_hinge" type="hinge" axis="{axis}" range="{jrange}" damping="0.5"/>
        <geom name="{n}_lid_geom" type="box" size="{child_size}" rgba="0.3 0.3 0.35 1" mass="0.05"/>
      </body>"""
    if "slide" in spec:
        sl = spec["slide"]
        axis = " ".join(str(x) for x in sl.get("axis", [1, 0, 0]))
        jrange = " ".join(str(x) for x in sl.get("range", [0, 0.25]))
        handle_pos = " ".join(f"{x:.4f}" for x in sl.get("pos", [0.1, 0, 0]))
        handle_size = " ".join(f"{x:.4f}" for x in sl.get("size", [0.02, 0.04, 0.02]))
        extra = f"""
      <body name="{n}_slider" pos="{handle_pos}">
        <joint name="{n}_slide" type="slide" axis="{axis}" range="{jrange}" damping="1"/>
        <geom name="{n}_handle" type="box" size="{handle_size}" rgba="0.6 0.4 0.2 1" mass="0.08"/>
      </body>"""

    return f"""
    <body name="{n}" pos="{pos_s}" quat="{quat_s}">
      <freejoint name="{n}_free"/>
      {geom}
      {extra}
    </body>"""


def build_scene_xml(
    task: TaskSpec,
    robot_include: str = "g1_with_hands.xml",
    table_pos: Sequence[float] = (0.55, 0.0, 0.75),
    table_size: Sequence[float] = (0.4, 0.5, 0.02),
    bin_pos: Sequence[float] = (0.55, -0.25, 0.78),
) -> str:
    """Compose a MuJoCo scene that includes the official robot XML + tabletop props.

    The scene file must live next to ``g1_with_hands.xml`` so the relative include
    and ``meshdir="assets"`` resolve (same pattern as Menagerie ``scene_with_hands.xml``).
    """
    objects_xml = "\n".join(_object_body_xml(n, s) for n, s in task.objects.items())
    tx, ty, tz = table_pos
    tsx, tsy, tsz = table_size
    bx, by, bz = bin_pos

    # Soft weld: world anchor at stand pelvis height, welded to pelvis.
    return f"""<mujoco model="g1_he_{_xml_escape(task.name)}">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <include file="{robot_include}"/>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.2 0.25 0.2 1" friction="1 0.005 0.0001"/>

    <!-- Table -->
    <body name="table" pos="{tx} {ty} {tz}">
      <geom name="table_top" type="box" size="{tsx} {tsy} {tsz}" rgba="0.55 0.4 0.25 1" friction="1.2 0.01 0.0001"/>
      <geom name="table_leg_fl" type="cylinder" size="0.03 0.36" pos="{-tsx+0.05} {-tsy+0.05} -0.38" rgba="0.3 0.3 0.3 1"/>
      <geom name="table_leg_fr" type="cylinder" size="0.03 0.36" pos="{-tsx+0.05} {tsy-0.05} -0.38" rgba="0.3 0.3 0.3 1"/>
      <geom name="table_leg_bl" type="cylinder" size="0.03 0.36" pos="{tsx-0.05} {-tsy+0.05} -0.38" rgba="0.3 0.3 0.3 1"/>
      <geom name="table_leg_br" type="cylinder" size="0.03 0.36" pos="{tsx-0.05} {tsy-0.05} -0.38" rgba="0.3 0.3 0.3 1"/>
    </body>

    <!-- Target bin (open box) -->
    <body name="bin" pos="{bx} {by} {bz}">
      <geom name="bin_floor" type="box" size="0.1 0.1 0.005" rgba="0.15 0.15 0.2 1"/>
      <geom name="bin_w1" type="box" size="0.1 0.005 0.06" pos="0 -0.1 0.06" rgba="0.2 0.2 0.3 1"/>
      <geom name="bin_w2" type="box" size="0.1 0.005 0.06" pos="0 0.1 0.06" rgba="0.2 0.2 0.3 1"/>
      <geom name="bin_w3" type="box" size="0.005 0.1 0.06" pos="-0.1 0 0.06" rgba="0.2 0.2 0.3 1"/>
      <geom name="bin_w4" type="box" size="0.005 0.1 0.06" pos="0.1 0 0.06" rgba="0.2 0.2 0.3 1"/>
      <site name="bin_site" pos="0 0 0.05" size="0.02" rgba="0 1 0 0.3"/>
    </body>

    <!-- Soft upright support anchor (elastic-band style) -->
    <body name="pelvis_anchor" pos="0 0 0.79">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
    </body>

    <!-- Egocentric camera near head / torso (HE-like FoV proxy) -->
    <camera name="egocentric" pos="0.08 0 1.45" xyaxes="0 -1 0 0.5 0 0.866" fovy="70"/>

    {objects_xml}
  </worldbody>

  <equality>
    <weld name="pelvis_support" body1="pelvis" body2="pelvis_anchor"
          solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </equality>
</mujoco>
"""


class G1MujocoEnv:
    """Minimal G1 tabletop env driven by 28-D absolute joint targets."""

    def __init__(
        self,
        robot_xml: Optional[Path] = None,
        control_hz: float = 20.0,
        image_height: int = 480,
        image_width: int = 640,
        render_third_person: bool = False,
    ):
        import mujoco

        self.mujoco = mujoco
        self.robot_xml = Path(robot_xml) if robot_xml else default_robot_xml()
        # Ensure compiler meshdir resolves: load from robot dir context via include abs path.
        self.control_hz = control_hz
        self.image_height = image_height
        self.image_width = image_width
        self.render_third_person = render_third_person

        self.model: Any = None
        self.data: Any = None
        self.renderer: Any = None
        self._task: Optional[TaskSpec] = None
        self._scene_path: Optional[Path] = None
        self._n_substeps = 1
        self._policy_act_ids: List[int] = []
        self._held_act_ids: List[int] = []
        self._held_targets: Dict[str, float] = {}
        self._stand_ctrl: Optional[np.ndarray] = None
        self._step_count = 0

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self._scene_path is not None and self._scene_path.is_file():
            try:
                self._scene_path.unlink()
            except OSError:
                pass
            self._scene_path = None

    def reset(self, task: TaskSpec, seed: int = 0) -> Dict[str, Any]:
        mujoco = self.mujoco
        rng = np.random.default_rng(seed)
        self.close()
        # Scene must sit next to g1_with_hands.xml so meshdir="assets" resolves
        # (same pattern as Menagerie scene_with_hands.xml).
        robot_dir = menagerie_unitree_g1_dir()
        scene_path = robot_dir / f".he_eval_{task.name}_{os.getpid()}_{seed}.xml"
        # Small init noise on free objects
        objects = {}
        for name, spec in task.objects.items():
            s = dict(spec)
            pos = list(s.get("pos", [0.45, 0.0, 0.85]))
            pos[0] += float(rng.uniform(-0.02, 0.02))
            pos[1] += float(rng.uniform(-0.02, 0.02))
            s["pos"] = pos
            objects[name] = s
        noisy = TaskSpec(
            name=task.name,
            suite=task.suite,
            instruction=task.instruction,
            max_steps=task.max_steps,
            objects=objects,
            success_type=task.success_type,
            success_params=dict(task.success_params),
        )
        scene_path.write_text(
            build_scene_xml(noisy, robot_include="g1_with_hands.xml")
        )
        self._scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self._task = noisy
        self._step_count = 0

        dt = self.model.opt.timestep
        self._n_substeps = max(1, int(round((1.0 / self.control_hz) / dt)))

        # Resolve actuator ids by joint name (Menagerie names actuators = joints).
        name_to_act = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                name_to_act[name] = i
        missing = [j for j in POLICY_JOINTS if j not in name_to_act]
        if missing:
            raise RuntimeError(f"policy joints missing as actuators: {missing}")
        self._policy_act_ids = [name_to_act[j] for j in POLICY_JOINTS]
        self._held_act_ids = [name_to_act[j] for j in LEG_WAIST_JOINTS if j in name_to_act]

        # Stand keyframe if present.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        # Clamp ctrl into actuator ranges.
        for i in range(self.model.nu):
            lo, hi = self.model.actuator_ctrlrange[i]
            if hi > lo:
                self.data.ctrl[i] = float(np.clip(self.data.ctrl[i], lo, hi))
        self._stand_ctrl = self.data.ctrl.copy()
        self._held_targets = {
            LEG_WAIST_JOINTS[k]: float(self._stand_ctrl[self._held_act_ids[k]])
            for k in range(len(self._held_act_ids))
        }

        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(
            self.model, height=self.image_height, width=self.image_width
        )
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        assert self.model is not None and self.renderer is not None
        mujoco = self.mujoco
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "egocentric")
        self.renderer.update_scene(self.data, camera=cam_id)
        rgb = self.renderer.render()
        return {
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "instruction": self._task.instruction if self._task else "",
            "qpos": self.data.qpos.copy(),
            "ctrl": self.data.ctrl.copy(),
            "step": self._step_count,
        }

    def step(self, action28: Sequence[float]) -> Tuple[Dict[str, Any], float, bool, dict]:
        assert self.model is not None and self._task is not None
        mujoco = self.mujoco
        a = np.asarray(action28, dtype=np.float64).reshape(-1)
        if a.shape[0] != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}-D action, got {a.shape}")

        # Hold legs/waist at stand targets.
        for act_id in self._held_act_ids:
            self.data.ctrl[act_id] = self._stand_ctrl[act_id]

        # Drive policy joints.
        for i, act_id in enumerate(self._policy_act_ids):
            lo, hi = self.model.actuator_ctrlrange[act_id]
            tgt = float(a[i])
            if hi > lo:
                tgt = float(np.clip(tgt, lo, hi))
            self.data.ctrl[act_id] = tgt

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self.get_obs()
        success = self.check_success()
        terminated = bool(success) or self._step_count >= self._task.max_steps
        info = {"success": bool(success), "task": self._task.name, "suite": self._task.suite}
        reward = 1.0 if success else 0.0
        return obs, reward, terminated, info

    def body_pos(self, name: str) -> np.ndarray:
        mujoco = self.mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(name)
        return self.data.xpos[bid].copy()

    def joint_qpos(self, name: str) -> float:
        mujoco = self.mujoco
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(name)
        adr = self.model.jnt_qposadr[jid]
        return float(self.data.qpos[adr])

    def check_success(self) -> bool:
        assert self._task is not None
        t = self._task
        p = t.success_params
        st = t.success_type
        try:
            if st == "object_in_bin":
                obj = p.get("object", "cube")
                thresh = float(p.get("thresh", 0.12))
                return float(np.linalg.norm(self.body_pos(obj) - self.body_pos("bin"))) < thresh
            if st == "object_near_target":
                obj = p["object"]
                target = np.asarray(p["target_pos"], dtype=np.float64)
                thresh = float(p.get("thresh", 0.08))
                return float(np.linalg.norm(self.body_pos(obj) - target)) < thresh
            if st == "stack":
                top, base = p["top"], p["base"]
                xy = float(p.get("xy_thresh", 0.05))
                z_min = float(p.get("z_min", 0.03))
                d = self.body_pos(top) - self.body_pos(base)
                return abs(d[0]) < xy and abs(d[1]) < xy and d[2] > z_min
            if st == "align_xy":
                a, b = p["a"], p["b"]
                thresh = float(p.get("thresh", 0.04))
                da = self.body_pos(a)
                db = self.body_pos(b)
                # Same y-line (or x): both near a target line value.
                axis = p.get("axis", "y")
                idx = 0 if axis == "x" else 1
                return abs(da[idx] - db[idx]) < thresh
            if st == "joint_threshold":
                jn = p["joint"]
                vmin = float(p.get("min", 0.15))
                return abs(self.joint_qpos(jn)) >= vmin
            if st == "tilt_angle":
                # Success if object z-axis tilts enough (proxy pour).
                obj = p["object"]
                mujoco = self.mujoco
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj)
                # xmat is 3x3 row-major; column 2 is local z in world.
                z_world = self.data.xmat[bid].reshape(3, 3)[:, 2]
                upright = abs(float(z_world[2]))
                return upright < float(p.get("upright_max", 0.5))
            if st == "peg_in_hole":
                peg, hole = p["peg"], p["hole"]
                thresh = float(p.get("thresh", 0.04))
                return float(np.linalg.norm(self.body_pos(peg) - self.body_pos(hole))) < thresh
            if st == "contact_path":
                # Cloth/wipe: object traveled near a waypoint.
                obj = p["object"]
                wp = np.asarray(p["waypoint"], dtype=np.float64)
                return float(np.linalg.norm(self.body_pos(obj)[:2] - wp[:2])) < float(
                    p.get("thresh", 0.08)
                )
        except KeyError:
            return False
        return False
