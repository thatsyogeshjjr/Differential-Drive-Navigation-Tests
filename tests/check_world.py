"""Spawn rules over many seeds: clear of obstacles/walls, far from goal, >=1 obstacle blocks the
straight line, no contact after settling. Run: python tests/check_world.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from nav import config, world
from nav.sim import NavEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
for preset in (None, ROOT / "maps" / "spread_cubes.json"):
    cfg = config.load(preset)
    env, starts = NavEnv(cfg), set()
    for seed in range(150):
        env.reset(seed)
        w = env.world
        s, g = np.array(w["start"][:2]), np.array(w["goal"])
        starts.add(tuple(np.round(s, 3)))
        assert np.hypot(*(g - s)) >= cfg.world.min_start_goal, seed
        assert np.all(np.abs(s) < w["half"] - world.ROBOT_RADIUS), seed      # inside walls
        for _, x, y, r in w["obstacles"]:
            assert np.hypot(x - s[0], y - s[1]) > 1.42 * r + world.ROBOT_RADIUS, seed  # not on an obstacle
        assert world.blocking(w["start"], w["goal"], w["obstacles"]), seed
        assert not env._touching(), seed                                       # physics agrees
    assert len(starts) == 150, "starts should differ per seed"
    print(f"{preset.name if preset else 'default'}: 150 seeds OK, all starts distinct")
print("check_world OK")
