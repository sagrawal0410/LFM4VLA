"""Checkpoint policy server — runs under the LFM4VLA env (py3.10 + torch).

Protocol (JSON lines over stdio):
  in : {"instruction": str, "family": str, "frames": [b64 JPEG, ...]}
        frames are oldest->newest, at most window_size; the LAST is current.
  out: {"waypoints": [[x, y, yaw] x 8]}   # DENORMALIZED meters / radians
"""
import argparse
import base64
import io
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Reserve the real stdout for the JSON protocol; anything the model or its
    # libraries print (e.g. "Trainable Model Parameters: ...") goes to stderr.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    sys.path.insert(0, ".")
    import torch
    from PIL import Image

    variant = json.load(open(args.config))
    variant["trainer"]["logger"] = False
    variant.pop("resume", None)
    from train.robotnav_trainer import RobotNavTrainer
    module = RobotNavTrainer(variant)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    module.load_state_dict(ck["state_dict"], strict=False)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    module = module.to(device).eval()
    if device == "cpu":
        module = module.float()

    # frozen per-family denormalization (same file the dataset uses)
    rel = variant["train_dataset"]["release_dir"]
    txt = open(f"{rel}/scale_factors.yaml").read()
    scales = json.loads(txt[txt.index("{"):])

    ws = int(variant["window_size"])
    K = int(variant["fwd_pred_next_n"])
    proto.write("READY\n")
    proto.flush()

    for line in sys.stdin:
        try:
            req = json.loads(line)
            pils = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
                    for b in req["frames"]]
            while len(pils) < ws:                    # left-pad with earliest
                pils.insert(0, pils[0])
            pils = pils[-ws:]
            frames = [torch.from_numpy(
                __import__("numpy").array(p)).permute(2, 0, 1).contiguous()
                for p in pils]
            fam = req.get("family", "vln_r2r")
            sf = scales.get(fam) or scales["vln_r2r"]
            batch = {
                "rgb": [frames],                     # list-form (native res)
                "hand_rgb": None,
                "action": torch.zeros(1, ws, 3),
                "text": [req["instruction"]],
                "text_mask": None,
                "action_chunck": torch.zeros(1, ws, K, 3),
                "chunck_mask": torch.zeros(1, ws, K),
                "raw_text": [req["instruction"]],
                "data_source": "robotnav_traj",
                "family": [fam],
                "wp_scale": torch.tensor([[sf["x"], sf["y"], sf["yaw"]]]),
            }
            with torch.no_grad():
                pred = module._predict_waypoints(batch)   # [1, K, 3] normalized
            w = pred[0].cpu().float().numpy()
            w[:, 0] *= sf["x"]
            w[:, 1] *= sf["y"]
            w[:, 2] *= sf["yaw"]
            proto.write(json.dumps({"waypoints": w.tolist()}) + "\n")
            proto.flush()
        except Exception as e:  # noqa: BLE001
            proto.write(json.dumps({"error": repr(e)}) + "\n")
            proto.flush()


if __name__ == "__main__":
    main()
