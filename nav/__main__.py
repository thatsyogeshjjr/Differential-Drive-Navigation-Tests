"""CLI.  python -m nav {run,collect,train,eval} --help"""
import argparse
import json
import random
import time

from . import config, controllers
from .evaluate import evaluate, format_table, run_episode
from .sim import NavEnv


def _params(kvs):
    return dict(kv.split("=", 1) for kv in kvs or [])


class _Viewer:
    """Passive MuJoCo viewer, reopened per episode (each episode compiles a new arena)."""

    def __init__(self):
        import mujoco.viewer
        self.mv, self.handle, self.model = mujoco.viewer, None, None

    def __call__(self, env):
        if env.model is not self.model:
            if self.handle:
                self.handle.close()
            self.model, self.handle = env.model, self.mv.launch_passive(env.model, env.data)
        self.handle.sync()
        time.sleep(1 / env.cfg.episode.control_hz)


def main(argv=None):
    p = argparse.ArgumentParser(prog="nav")
    p.add_argument("--config", help="JSON file overriding nav/config.py defaults")
    sub = p.add_subparsers(dest="cmd", required=True)

    def ctrl_args(sp, multi=False):
        if multi:
            sp.add_argument("--controllers", nargs="+", default=None, help=f"default: all ({', '.join(controllers.names())})")
        else:
            sp.add_argument("--controller", "-c", required=True, choices=controllers.names())
        sp.add_argument("-p", "--param", action="append", metavar="KEY=VALUE", help="controller parameter, e.g. -p model=models/bc.npz")

    r = sub.add_parser("run", help="run episode(s) with one controller")
    ctrl_args(r)
    r.add_argument("--seed", type=int, default=None, help="default: random (printed in the result)")
    r.add_argument("--episodes", type=int, default=1)
    r.add_argument("--render", action="store_true")

    c = sub.add_parser("collect", help="record demonstrations to a dataset")
    ctrl_args(c)
    c.add_argument("--episodes", type=int, default=100)
    c.add_argument("--seed", type=int, default=10000, help="first seed (keep disjoint from eval seeds)")
    c.add_argument("--out", required=True)

    t = sub.add_parser("train", help="fit the supervised (behaviour cloning) controller")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", default="models/bc.npz")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("eval", help="compare controllers on identical seeds")
    ctrl_args(e, multi=True)
    e.add_argument("--episodes", type=int, default=50)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", help="per-episode CSV")
    e.add_argument("--json", help="summary JSON")

    a = p.parse_args(argv)
    cfg = config.load(a.config)

    if a.cmd in ("run", "collect"):
        recorder = None
        if a.cmd == "collect":
            from .dataset import Recorder
            recorder = Recorder(a.out, a.controller, config.to_dict(cfg))
        ctrl, env = controllers.make(a.controller, cfg, **_params(a.param)), NavEnv(cfg)
        viewer = _Viewer() if getattr(a, "render", False) else None
        if a.seed is None:
            a.seed = random.randrange(10 ** 6)
        for s in range(a.seed, a.seed + a.episodes):
            print(json.dumps(run_episode(env, ctrl, s, recorder, viewer)))
    elif a.cmd == "train":
        from .train import train
        train(a.data, a.out, epochs=a.epochs, hidden=a.hidden, seed=a.seed)
    elif a.cmd == "eval":
        names = a.controllers or controllers.names()
        table = evaluate(cfg, names, range(a.seed, a.seed + a.episodes), _params(a.param), a.out)
        print(format_table(table))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
