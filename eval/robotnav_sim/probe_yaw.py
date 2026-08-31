"""One-off probe: full predicted waypoint plans (x, y, yaw) at step 0 of a
VLN episode, for several checkpoints. Run under the habitat env.

  python -m eval.robotnav_sim.probe_yaw --episode 1 \
      --models "name1:config1:ckpt1,name2:config2:ckpt2" [--draws 3]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from PIL import Image

from eval.robotnav_sim import core, run as runmod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="1")
    ap.add_argument("--suite", default="vlnce_r2r")
    ap.add_argument("--split", default="val_seen")
    ap.add_argument("--models", required=True)
    ap.add_argument("--draws", type=int, default=1)
    args = ap.parse_args()

    pool = runmod.load_vlnce(args.suite, args.split, 10 ** 6)
    ep = next(e for e in pool if e["id"] == args.episode)
    print(f"[episode {ep['id']}] scene={ep['scene']}")
    print(f"instr: {ep['instruction']}")

    import quaternion  # noqa: F401
    sim = core.make_eval_sim(ep["scene_glb"], turn_deg=ep["turn_deg"])
    rot = ep["start_rot"]
    q = np.quaternion(rot[3], rot[0], rot[1], rot[2])
    core.set_agent(sim, ep["start_pos"], q)
    st = sim.get_agent(0).get_state()
    pd = core.next_path_dir(sim, st, ep["goal"])
    if pd:
        print(f"geodesic next-dir: fwd={pd[0]:+.2f} left={pd[1]:+.2f} "
              f"(bearing {math.degrees(math.atan2(pd[1], pd[0])):+.0f} deg)")
    obs = sim.get_sensor_observations()["front"][..., :3]
    frame = Image.fromarray(obs)
    sim.close()

    for spec in args.models.split(","):
        name, config, ckpt = spec.strip().split(":")
        client = core.PolicyClient(config, ckpt, device="cpu")
        for d in range(args.draws):
            wps = client.predict(ep["instruction"], [frame], ep["family"])
            trans = float(np.linalg.norm(wps[-1, :2]))
            print(f"--- {name} draw {d}  |wp8_xy|={trans:.3f} m "
                  f"yaw8={math.degrees(float(wps[-1, 2])):+.1f} deg")
            for k in range(wps.shape[0]):
                print(f"    wp{k+1}: x={wps[k,0]:+.3f} y={wps[k,1]:+.3f} "
                      f"yaw={math.degrees(float(wps[k,2])):+.1f}deg")
        client.close()


if __name__ == "__main__":
    main()
