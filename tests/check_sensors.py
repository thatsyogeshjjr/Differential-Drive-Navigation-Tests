"""M2: odometry drift, IMU, depth camera geometry. Run: python tests/check_sensors.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from nav import config
from nav.sim import NavEnv, _quat


def empty_env(seed=0):
    cfg = config.load()
    cfg.world.n_obstacles, cfg.world.size = 0, 20.0
    env = NavEnv(cfg)
    env.reset(seed)
    return env


def gt_in_start_frame(env, p0):
    g = env.gt_pose()
    c, s = np.cos(p0[2]), np.sin(p0[2])
    dx, dy = g[:2] - p0[:2]
    return np.array([c * dx + s * dy, -s * dx + c * dy, (g[2] - p0[2] + np.pi) % (2 * np.pi) - np.pi])


# --- odometry: 1 m square, open loop ---
env = empty_env()
p0 = env.gt_pose()
for _ in range(4):
    for _ in range(67):
        obs, _, info = env.step(0.3, 0.0)
    for _ in range(32):
        obs, _, info = env.step(0.0, np.pi / 2 / 1.6)
gt = gt_in_start_frame(env, p0)
err = np.hypot(*(obs["odom"][:2] - gt[:2]))
yaw_err = abs((obs["odom"][2] - gt[2] + np.pi) % (2 * np.pi) - np.pi)
print(f"odometry after {info['path_length']:.2f} m: pos err {err:.3f} m, yaw err {np.degrees(yaw_err):.2f} deg")
assert err < 0.05 * info["path_length"] and yaw_err < np.radians(3)

# --- IMU ---
env = empty_env(1)
gz, wz = [], []
for _ in range(40):
    obs, _, _ = env.step(0.0, 1.0)
    gz.append(obs["imu"][2]); wz.append(env.data.qvel[5])
print(f"gyro z mean {np.mean(gz[10:]):.3f} vs GT {np.mean(wz[10:]):.3f} rad/s")
assert abs(np.mean(gz[10:]) - np.mean(wz[10:])) < 0.02
for _ in range(20):
    obs, _, _ = env.step(0.0, 0.0)
print(f"accel at rest {obs['imu'][3:].round(2)}")
assert abs(obs["imu"][5] - 9.81) < 0.2 and np.all(np.abs(obs["imu"][3:5]) < 0.2)


# --- depth: wall at x=10 (inner face), camera 2 m away ---
def place(env, x):
    env.data.qpos[:7] = [x, 0, 0, *_quat(0.0)]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    return env._observe()


cam_x = env.cam.pos[0]
obs = place(env, 10 - 2.0 - cam_x)
mid = obs["depth"][25:35, 30:50]
ahead = lambda d: (lambda p: p[np.abs(p[:, 1]).argmin(), 0])(env.cam.to_scan(d))
print(f"depth at wall 2.0 m: median {np.median(mid):.3f}, scan ahead x={ahead(obs['depth']):.3f} (expect {2.0 + cam_x:.3f})")
assert abs(np.median(mid) - 2.0) < 0.03 and abs(ahead(obs["depth"]) - (2.0 + cam_x)) < 0.05
floor = obs["depth"][-1, 40]
print(f"bottom row hits floor at z-depth {floor:.3f} (expect {0.3 / np.tan(np.radians(24.75)):.3f}); scan ignores floor")
obs = place(env, 10 - 0.4 - cam_x)
print(f"wall 0.4 m (< min range): centre depth {obs['depth'][30, 40]}, scan ahead x={ahead(obs['depth']):.3f} (reported at min range)")
assert obs["depth"][30, 40] == 0 and ahead(obs["depth"]) < 0.6

# --- bumper: drive into the wall ---
for _ in range(40):
    obs, _, info = env.step(0.2, 0.0)
print(f"bumper after driving into wall: {obs['bumper']}, collisions {info['collisions']}")
assert obs["bumper"][1] == 1 and info["collisions"] == 1
print("check_sensors OK")
