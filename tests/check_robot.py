"""M1: diff-drive kinematics match commands; robot stays upright. Run: python tests/check_robot.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from nav import config
from nav.sim import NavEnv


def empty_env():
    cfg = config.load()
    cfg.world.n_obstacles, cfg.world.size = 0, 20.0
    env = NavEnv(cfg)
    env.reset(0)
    return env


def upright(env):
    q = env.data.qpos[3:7]
    return 1 - 2 * (q[1] ** 2 + q[2] ** 2) > 0.999  # body z . world z


DT = 0.05  # expected motion = integral of the applied (velocity-smoothed) command
env = empty_env()
p0, expect = env.gt_pose(), 0.0
for _ in range(60):  # 3 s at 20 Hz
    env.step(0.3, 0.0)
    expect += env.cmd[0] * DT
d = np.hypot(*(env.gt_pose()[:2] - p0[:2]))
print(f"straight: {d:.3f} m (expect {expect:.3f})")
assert abs(d - expect) < 0.03 * expect and upright(env)

env = empty_env()
p0 = env.gt_pose()
yaw = expect = 0.0
for _ in range(60):
    prev = env.gt_pose()[2]
    env.step(0.0, 1.0)
    expect += env.cmd[1] * DT
    yaw += (env.gt_pose()[2] - prev + np.pi) % (2 * np.pi) - np.pi
drift = np.hypot(*(env.gt_pose()[:2] - p0[:2]))
print(f"spin: {yaw:.3f} rad (expect {expect:.3f}), centre drift {drift:.4f} m")
assert abs(yaw - expect) < 0.05 * expect and drift < 0.02 and upright(env)

env = empty_env()
for _ in range(40):
    env.step(0.7, 0.0)
v = np.hypot(*env.data.qvel[:2])
print(f"top speed: {v:.3f} m/s (expect 0.7)")
assert abs(v - 0.7) < 0.035 and upright(env)
print("check_robot OK")
