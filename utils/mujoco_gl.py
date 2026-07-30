"""Headless MuJoCo GL backend selection (EGL / software EGL / OSMesa).

Must run *before* ``import mujoco`` / ``import robosuite`` / ``import libero``.

Important: ``import mujoco`` alone is NOT a valid probe on SLURM — GPU EGL often
imports fine but fails when robosuite creates an OffScreen EGL context
(PLATFORM_DEVICE). We probe by creating a real offscreen context.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Prefer software / surfaceless EGL before raw GPU EGL — bare ``egl`` often
# passes ``import mujoco`` on SLURM then dies inside robosuite's EGLGLContext.
MUJOCO_GL_PROFILES: Tuple[Dict[str, str], ...] = (
    {
        "name": "egl_software",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBGL_ALWAYS_SOFTWARE": "true",
    },
    {
        "name": "egl_headless",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBGL_ALWAYS_SOFTWARE": "true",
        "EGL_PLATFORM": "surfaceless",
    },
    {"name": "osmesa", "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"},
    {"name": "egl", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
)

_PROBE_SCRIPT = r"""
import os
# 1) mujoco offscreen renderer (respects MUJOCO_GL)
import mujoco
xml = '''<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size="0.1"/></worldbody></mujoco>'''
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, height=64, width=64)
r.update_scene(d)
_ = r.render()
r.close()

# 2) robosuite EGL path (what OffScreenRenderEnv actually uses when MUJOCO_GL=egl)
gl = os.environ.get("MUJOCO_GL", "").lower()
if gl == "egl":
    from robosuite.renderers.context.egl_context import EGLGLContext
    ctx = EGLGLContext(max_width=64, max_height=64, device_id=-1)
    ctx.free()
elif gl == "osmesa":
    # OSMesa path is exercised by mujoco.Renderer above when MUJOCO_GL=osmesa.
    pass
print("PROBE_OK")
"""


def prepend_conda_lib() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib = os.path.join(conda_prefix, "lib")
    path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib not in path.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = lib + (os.pathsep + path if path else "")


def apply_profile(profile: Dict[str, str]) -> None:
    prepend_conda_lib()
    # Clear keys that previous profiles may have set.
    for key in (
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "LIBGL_ALWAYS_SOFTWARE",
        "EGL_PLATFORM",
    ):
        os.environ.pop(key, None)
    for key, value in profile.items():
        if key != "name":
            os.environ[key] = value


def _probe(profile: Dict[str, str]) -> Tuple[bool, str]:
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib = os.path.join(conda_prefix, "lib")
        path = env.get("LD_LIBRARY_PATH", "")
        if lib not in path.split(os.pathsep):
            env["LD_LIBRARY_PATH"] = lib + (os.pathsep + path if path else "")
    for key in (
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "LIBGL_ALWAYS_SOFTWARE",
        "EGL_PLATFORM",
    ):
        env.pop(key, None)
    for key, value in profile.items():
        if key != "name":
            env[key] = value

    result = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and "PROBE_OK" in (result.stdout or ""):
        return True, ""
    err = (result.stderr or result.stdout or "unknown error").strip()
    return False, err[-2000:]


def profiles_for_request(requested: Optional[str]) -> List[Dict[str, str]]:
    if requested in ("auto", "", None):
        return list(MUJOCO_GL_PROFILES)
    if requested == "egl":
        # Still try software variants first when user asks for egl.
        return [p for p in MUJOCO_GL_PROFILES if p["MUJOCO_GL"] == "egl"]
    if requested == "osmesa":
        return [p for p in MUJOCO_GL_PROFILES if p["MUJOCO_GL"] == "osmesa"]
    if requested in ("egl_software", "egl_headless"):
        return [p for p in MUJOCO_GL_PROFILES if p["name"] == requested]
    raise ValueError(f"unsupported mujoco_gl value: {requested}")


def configure_mujoco_gl(requested: str = "auto") -> str:
    """Pick a working headless backend; set env vars; return profile name."""
    profiles = profiles_for_request(requested)
    failures: List[str] = []
    for profile in profiles:
        print(f"[render] probing profile: {profile['name']} ...", flush=True)
        ok, err = _probe(profile)
        if ok:
            apply_profile(profile)
            print(f"[render] using MuJoCo GL profile: {profile['name']}", flush=True)
            return profile["name"]
        summary = err.splitlines()[-1] if err else "import failed"
        print(f"[render]   failed: {summary}", flush=True)
        failures.append(f"  {profile['name']}: {summary}")

    detail = "\n".join(failures)
    raise RuntimeError(
        "No headless MuJoCo GL backend available.\n"
        f"Probe results:\n{detail}\n\n"
        "Fix (inside your conda env on a compute node):\n"
        "  conda install -y -c conda-forge 'mesalib<=25.1.0' libegl-devel glew\n"
        "  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\n"
        "Then rerun with:  MUJOCO_GL=osmesa  or  MUJOCO_GL=egl_software\n"
    )
