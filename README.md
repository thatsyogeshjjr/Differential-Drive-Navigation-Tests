# Kobuki navigation testbed (MuJoCo)

TurtleBot 2 / Kobuki differential-drive robot in MuJoCo. Three interchangeable navigation controllers, a dataset recorder, and a reproducible evaluation harness.
Dependencies: `mujoco`, `numpy`.

```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
```

## Commands
```
python -m nav run      -c rule [--seed 3] [--episodes N] [--render]          # no --seed: random, printed
python -m nav collect  -c tsk  --episodes 120 --out data/tsk          # seeds 10000+ (disjoint from eval)
python -m nav train    --data data/tsk --out models/bc.npz
python -m nav eval     --controllers rule tsk learned --episodes 50 --out results/eval.csv --json results/summary.json
python -m nav eval     --controllers learned -p model=models/bc_rule.npz   # controller params via -p KEY=VALUE
python -m nav --config my.json eval ...                                # override any field of nav/config.py
```
Map presets live in `maps/`. For example, `python -m nav --config maps/spread_cubes.json run -c tsk --render` gives 7 cubes of 0.5 m, at least 0.9 m apart.
For example, `my.json` could be `{"world": {"n_obstacles": 12}, "sensors": {"imu": {"enabled": false}}}`.

## Layout
| file | role |
|---|---|
| `nav/kobuki.xml` | robot MJCF. Collision primitives are ported from `yujinrobot/kobuki_description` (the official meshes are `.dae`, which MuJoCo can't load), plus a lumped TB2 stack |
| `nav/world.py` | seeded arena: walls + random obstacles. Start/goal: clear of obstacles, ≥3.5 m apart, ≥1 obstacle blocking the straight line, reachable (grid BFS) |
| `nav/sim.py` | `NavEnv.reset(seed)` / `step(v, w)`: velocity smoother → wheel servos at 20 Hz, collisions, outcome |
| `nav/sensors.py` | `Encoders` (2578.33 ticks/rev), `Imu` (gyro+accel, noise, bias), `DepthCamera` (Astra: 60°×49.5°, 0.6–8 m, σ=1.425e-3·z²), `Bumper` (L/C/R), `Odometry` (encoder distance + gyro heading) |
| `nav/features.py` | depth scan → 3 s rolling local map → 19 features (goal range/bearing from **odometry**, front corridor, 12×30° clearances, bumpers) |
| `nav/controllers/` | `rule` (go-to-goal + Bug-style boundary following), `tsk` (first-order TSK fuzzy), `learned` (MLP behaviour cloning) |
| `nav/dataset.py` | per-episode `ep_<seed>.npz` (t, ticks, imu, depth, odom, features, gt_pose, gt_vel, cmd) + `index.jsonl` (outcome, world) + `config.json` |
| `nav/evaluate.py` | runs the same seeds for every controller. Reports success rate, collisions, path length, time |

**Adding a controller:** put a file in `nav/controllers/` containing `@register("name")` and a class with `act(obs) -> (v, w)`. It shows up in `--controller/--controllers` automatically.

## Physics notes (things that were wrong until fixed)
- Casters use `priority` so they stay frictionless; otherwise contact `condim` takes the max of both geoms. They sit 1 mm high because Kobuki's wheels are spring-loaded and carry the weight.
- Wheel `armature` (reflected gearmotor rotor inertia) is needed to keep the velocity servo numerically stable.
- An `elliptic` friction cone with `impratio=10` stops soft-contact creep. Without it the wheels slipped ~15% under braking and odometry was off by ~8%.
- Success is judged on ground truth. A robot that *believes* it has arrived (odometry within 5 cm) ends the episode, so odometry drift counts as a failure (`odom_arrived_off_goal`).

## Checks
```
python tests/check_robot.py      # kinematics track commands; upright; top speed
python tests/check_sensors.py    # odometry drift, gyro/accel, depth geometry & min range, bumper
python tests/check_pipeline.py   # registry/extensibility, bit-exact determinism, dataset sync
python tests/check_world.py      # spawn rules over 150 seeds per map
```

## Not included
RGB stream, ROS bridge, GPU depth rendering (ray casting is headless and deterministic), DAgger for the learned controller (BC from the stateless TSK clones better than from the stateful rule controller).
