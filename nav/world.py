"""Seeded arena generation: walls + random obstacles, guaranteed-reachable start/goal."""
from collections import deque
from pathlib import Path

import numpy as np

ROBOT_XML = (Path(__file__).parent / "kobuki.xml").read_bytes()
ROBOT_RADIUS = 0.178


def _reachable(start, goal, obstacles, half, inflate, res=0.05):
    """Grid BFS on the inflated map, so every generated episode is solvable."""
    n = int(2 * half / res)
    c = (np.arange(n) + 0.5) * res - half
    X, Y = np.meshgrid(c, c, indexing="ij")
    free = (np.abs(X) < half - inflate) & (np.abs(Y) < half - inflate)
    for kind, x, y, r in obstacles:  # box half-size r ~ conservative circle r*sqrt2
        rr = r * (1.42 if kind == "box" else 1.0) + inflate
        free &= (X - x) ** 2 + (Y - y) ** 2 > rr ** 2
    idx = lambda p: tuple(np.clip(((np.asarray(p) + half) / res).astype(int), 0, n - 1))
    s, g = idx(start), idx(goal)
    if not (free[s] and free[g]):
        return False
    seen = np.zeros_like(free)
    seen[s] = True
    q = deque([s])
    while q:
        i, j = q.popleft()
        if (i, j) == g:
            return True
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < n and 0 <= b < n and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return False


def blocking(start, goal, obstacles):
    """Obstacles the robot would hit driving straight from start to goal (inscribed radius, so strict)."""
    a, d = np.asarray(start[:2]), np.asarray(goal) - start[:2]
    out = []
    for o in obstacles:
        p = np.array(o[1:3])
        t = np.clip((p - a) @ d / (d @ d), 0, 1)
        if np.hypot(*(p - a - t * d)) < o[3] + ROBOT_RADIUS:
            out.append(o)
    return out


def generate(wcfg, rng):
    half = wcfg.size / 2
    margin = half - 0.5
    while True:
        start = rng.uniform(-margin, margin, 2)
        goal = rng.uniform(-margin, margin, 2)
        if np.linalg.norm(goal - start) < wcfg.min_start_goal:
            continue
        obstacles, tries = [], 0
        while len(obstacles) < wcfg.n_obstacles:
            tries += 1
            if tries > 20000:
                raise ValueError("obstacles don't fit: lower world.n_obstacles or world.min_gap")
            x, y = rng.uniform(-half + 0.3, half - 0.3, 2)
            r = rng.uniform(*wcfg.obstacle_radius)
            kind = "box" if rng.random() < wcfg.box_fraction else "cyl"
            if wcfg.min_gap > 0 and any(np.hypot(x - o[1], y - o[2]) < 1.42 * (r + o[3]) + wcfg.min_gap for o in obstacles):
                continue
            if min(np.hypot(x - p[0], y - p[1]) for p in (start, goal)) > r * 1.42 + ROBOT_RADIUS + 0.3:
                obstacles.append((kind, float(x), float(y), float(r)))
        if (blocking(start, goal, obstacles) or not obstacles) and _reachable(start, goal, obstacles, half, ROBOT_RADIUS + 0.07):
            yaw = rng.uniform(-np.pi, np.pi)
            return dict(start=(*start, yaw), goal=tuple(goal), obstacles=obstacles, half=half)


def to_mjcf(world, timestep):
    h, t = world["half"], 0.05
    geoms = [f'<geom name="wall{i}" type="box" size="{sx} {sy} 0.25" pos="{x} {y} 0.25" rgba=".6 .5 .4 1"/>'
             for i, (x, y, sx, sy) in enumerate([(h + t, 0, t, h + 2 * t), (-h - t, 0, t, h + 2 * t),
                                                  (0, h + t, h, t), (0, -h - t, h, t)])]
    for i, (kind, x, y, r) in enumerate(world["obstacles"]):
        size = f"{r} {r} 0.25" if kind == "box" else f"{r} 0.25"
        typ = "box" if kind == "box" else "cylinder"
        geoms.append(f'<geom name="obs{i}" type="{typ}" size="{size}" pos="{x} {y} 0.25" rgba=".8 .3 .2 1"/>')
    gx, gy = world["goal"]
    return f"""<mujoco model="arena">
  <!-- elliptic cone + impratio: removes soft-contact friction creep that otherwise makes
       wheels slip ~15% while braking and corrupts odometry -->
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <include file="kobuki.xml"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="{h + 1} {h + 1} 0.1" rgba=".9 .9 .9 1"/>
    {chr(10).join(geoms)}
    <site name="goal" pos="{gx} {gy} 0.01" size="0.1 0.005" rgba="0 .8 0 .6" type="cylinder"/>
  </worldbody>
</mujoco>"""
