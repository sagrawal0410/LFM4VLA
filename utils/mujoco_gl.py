"""Headless MuJoCo GL backend selection (EGL / software EGL / OSMesa).

Must run *before* ``import mujoco`` / ``import robosuite`` / ``import libero``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Tuple

# Tried in order when requested="auto".
MUJOCO_GL_PROFILES: Tuple[Dict[str, str], ...] = (
    {"name": "egl", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
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
)


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
    for key, value in profile.items():
        if key != "name":
            env[key] = value
    result = subprocess.run(
        [sys.executable, "-c", "import mujoco"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, ""
    err = (result.stderr or result.stdout or "unknown error").strip()
    return False, err[-1500:]


def profiles_for_request(requested: str) -> List[Dict[str, str]]:
    if requested in ("auto", "", None):
        profiles = list(MUJOCO_GL_PROFILES)
        env_pref = os.environ.get("MUJOCO_GL")
        if env_pref and env_pref not in ("auto", ""):
            preferred = [p for p in profiles if p["MUJOCO_GL"] == env_pref]
            others = [p for p in profiles if p["MUJOCO_GL"] != env_pref]
            return preferred + others
        return profiles
    if requested == "egl":
        return [p for p in MUJOCO_GL_PROFILES if p["MUJOCO_GL"] == "egl"]
    if requested == "osmesa":
        return [p for p in MUJOCO_GL_PROFILES if p["MUJOCO_GL"] == "osmesa"]
    raise ValueError(f"unsupported mujoco_gl value: {requested}")


def configure_mujoco_gl(requested: str = "auto") -> str:
    """Pick a working headless backend; set env vars; return profile name."""
    profiles = profiles_for_request(requested)
    failures: List[str] = []
    for profile in profiles:
        ok, err = _probe(profile)
        if ok:
            apply_profile(profile)
            print(f"[render] using MuJoCo GL profile: {profile['name']}", flush=True)
            return profile["name"]
        summary = err.splitlines()[-1] if err else "import failed"
        failures.append(f"  {profile['name']}: {summary}")

    detail = "\n".join(failures)
    raise RuntimeError(
        "No headless MuJoCo GL backend available.\n"
        f"Probe results:\n{detail}\n\n"
        "Fix (inside your conda env on a compute node):\n"
        "  conda install -y -c conda-forge mesalib libegl-devel glew\n"
        "  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\n"
        "If osmesa fails on newer mesalib: conda install -c conda-forge 'mesalib<=25.1.0'\n"
    )
