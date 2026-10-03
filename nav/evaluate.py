"""Episode runner + metrics. Same seeds => same worlds and sensor noise for every controller."""
import csv
import time

import numpy as np

from . import controllers
from .sim import NavEnv


def run_episode(env, ctrl, seed, recorder=None, viewer=None):
    obs = env.reset(seed)
    ctrl.reset()
    if recorder:
        recorder.start(seed, env)
    t0, done = time.perf_counter(), False
    while not done:
        gt, gv = env.gt_pose(), env.data.qvel[[0, 1, 5]].copy()
        v, w = ctrl.act(obs)
        if recorder:
            recorder.log(obs, gt, gv, (v, w))
        obs, done, info = env.step(v, w)
        if viewer:
            viewer(env)
    result = dict(seed=seed, success=bool(info["success"]), outcome=info["outcome"], collisions=int(info["collisions"]),
                  path_length=round(float(info["path_length"]), 4), time=round(float(env.data.time), 3),
                  final_goal_dist=round(float(info["goal_dist"]), 4),
                  wall_clock=round(time.perf_counter() - t0, 3))
    if recorder:
        recorder.finish(result)
    return result


def summarize(rows):
    ok = [r for r in rows if r["success"]]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return dict(episodes=len(rows), success_rate=len(ok) / len(rows),
                collision_free_success=sum(r["collisions"] == 0 for r in ok) / len(rows),
                collisions_mean=mean([r["collisions"] for r in rows]),
                path_length_success=mean([r["path_length"] for r in ok]),
                time_success=mean([r["time"] for r in ok]),
                wall_clock_mean=mean([r["wall_clock"] for r in rows]))


def evaluate(cfg, names, seeds, params=None, out_csv=None, log=print):
    all_rows, table = [], {}
    for name in names:
        ctrl, env = controllers.make(name, cfg, **(params or {})), NavEnv(cfg)
        rows = []
        for s in seeds:
            r = run_episode(env, ctrl, s)
            rows.append(dict(controller=name, **r))
            log(f"  {name:>8} seed {s:>6}: {r['outcome']:<22} col={r['collisions']} "
                f"len={r['path_length']:.2f} t={r['time']:.1f}")
        table[name] = summarize(rows)
        all_rows += rows
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    return table


def format_table(table):
    cols = ["success_rate", "collision_free_success", "collisions_mean", "path_length_success", "time_success", "wall_clock_mean"]
    lines = [f"{'controller':>10} " + " ".join(f"{c:>22}" for c in cols)]
    for name, s in table.items():
        lines.append(f"{name:>10} " + " ".join(f"{s[c]:>22.3f}" for c in cols))
    return "\n".join(lines)
