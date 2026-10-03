"""Episode recorder: synchronised per-step arrays + outcome -> <out>/ep_<seed>.npz, index.jsonl.

Step k holds the observation at time t[k], the ground-truth state at that same instant,
and the command the controller issued in response to it.
"""
import json
from pathlib import Path

import numpy as np

from . import features


class Recorder:
    def __init__(self, out_dir, controller, cfg_dict):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.controller, self.cfg_dict = controller, cfg_dict

    def start(self, seed, env):
        self.seed, self.rows = seed, []
        self.meta = dict(start=env.world["start"], goal_world=env.world["goal"], goal_odom=env.goal_odom.tolist(),
                         obstacles=env.world["obstacles"])

    def log(self, obs, gt_pose, gt_vel, cmd):
        self.rows.append((obs, gt_pose, gt_vel, cmd))

    def finish(self, result):
        get = lambda k: np.array([r[0][k] for r in self.rows])
        arrays = dict(t=get("t"), ticks=get("ticks"), odom=get("odom"), features=get("features"),
                      gt_pose=np.array([r[1] for r in self.rows]), gt_vel=np.array([r[2] for r in self.rows]),
                      cmd=np.array([r[3] for r in self.rows], np.float32))
        if self.rows[0][0]["imu"] is not None:
            arrays["imu"] = get("imu")
        if self.rows[0][0]["depth"] is not None:
            arrays["depth"] = get("depth").astype(np.float16)
        path = self.out / f"ep_{self.seed:06d}.npz"
        np.savez_compressed(path, **arrays)
        entry = dict(file=path.name, controller=self.controller, **result, **self.meta,
                     feature_names=features.names())
        with open(self.out / "index.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        (self.out / "config.json").write_text(json.dumps(self.cfg_dict, indent=1))


def load(dirs, successful_only=True):
    """Concatenate (features, cmd) from one or more recorded directories."""
    X, Y = [], []
    for d in map(Path, dirs):
        for line in open(d / "index.jsonl"):
            e = json.loads(line)
            if successful_only and not e["success"]:
                continue
            with np.load(d / e["file"]) as z:
                X.append(z["features"]); Y.append(z["cmd"])
    return np.concatenate(X), np.concatenate(Y)
