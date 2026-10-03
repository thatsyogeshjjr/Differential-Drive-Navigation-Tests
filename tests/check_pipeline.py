"""Registry, extensibility, determinism, dataset synchronisation. Run: python tests/check_pipeline.py [data_dir]"""
import json, sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from nav import config, controllers
from nav.controllers import register
from nav.controllers.base import Controller
from nav.dataset import Recorder
from nav.evaluate import evaluate, run_episode
from nav.sim import NavEnv

assert {"rule", "tsk", "learned"} <= set(controllers.names()), controllers.names()


@register("spin_test")  # a new controller needs nothing but this
class Spin(Controller):
    def act(self, obs):
        return 0.0, 0.5


cfg = config.load()
cfg.episode.max_time = 2.0
table = evaluate(cfg, ["spin_test"], [0], log=lambda *_: None)
assert table["spin_test"]["episodes"] == 1
print("extensibility: ad-hoc controller evaluated without touching the package")


def traj(seed):
    env, c = NavEnv(config.load()), controllers.make("rule", config.load())
    obs, done, out = env.reset(seed), False, []
    while not done:
        obs, done, _ = env.step(*c.act(obs))
        out.append(np.r_[env.data.qpos, obs["features"]])
    return np.array(out)


assert np.array_equal(traj(3), traj(3))
print("determinism: identical seed -> bit-identical trajectory")

with tempfile.TemporaryDirectory() as d:
    cfg = config.load()
    rec = Recorder(d, "rule", config.to_dict(cfg))
    res = run_episode(NavEnv(cfg), controllers.make("rule", cfg), 7, rec)
    with np.load(pathlib.Path(d) / "ep_000007.npz") as f:
        z = dict(f)
    n = len(z["t"])
    assert all(len(z[k]) == n for k in z), {k: z[k].shape for k in z}
    assert np.allclose(np.diff(z["t"]), 1 / cfg.episode.control_hz)
    assert z["depth"].shape[1:] == (cfg.sensors.depth.height, cfg.sensors.depth.width)
    e = json.loads((pathlib.Path(d) / "index.jsonl").read_text())
    assert e["success"] == res["success"] and e["seed"] == 7
    # recorded GT pose at step 0 equals the generated start pose (sync check)
    assert np.allclose(z["gt_pose"][0][:2], e["start"][:2], atol=1e-3)
    print(f"dataset: {n} synchronised steps, keys {sorted(z)}")
print("check_pipeline OK")
