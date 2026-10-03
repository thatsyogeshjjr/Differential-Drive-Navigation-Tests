"""Simulated Kobuki/TurtleBot 2 sensors. Each takes its own config + rng and reads MjData."""
import mujoco
import numpy as np

ROBOT_GROUP = 2  # robot geoms live in this geom group (see kobuki.xml)


class Encoders:
    """Wheel encoders: integer ticks, quantised like Kobuki's 2578.33 ticks/rev."""

    def __init__(self, cfg, rcfg, model, rng):
        self.tpr = rcfg.ticks_per_rev
        self.adr = [model.sensor_adr[model.sensor(n).id] for n in ("enc_l", "enc_r")]

    def read(self, data):
        ang = data.sensordata[self.adr]
        return np.floor(ang / (2 * np.pi) * self.tpr).astype(np.int64)


class Imu:
    """Gyro (rad/s) + accelerometer (m/s^2) in the IMU frame, with white noise and constant gyro bias.

    Averages over every physics step of the control period (anti-aliasing, as real IMUs do);
    call accumulate() once per mj_step.
    """

    def __init__(self, cfg, rcfg, model, rng):
        self.cfg, self.rng = cfg, rng
        self.g = model.sensor_adr[model.sensor("gyro").id]
        self.a = model.sensor_adr[model.sensor("accel").id]
        self.bias = rng.normal(0, cfg.gyro_bias_std, 3)
        self._sum, self._n = np.zeros(6), 0

    def accumulate(self, data):
        self._sum += np.concatenate([data.sensordata[self.g:self.g + 3], data.sensordata[self.a:self.a + 3]])
        self._n += 1

    def read(self, data):
        if self._n == 0:
            self.accumulate(data)
        raw, self._sum, self._n = self._sum / self._n, np.zeros(6), 0
        gyro = raw[:3] + self.bias + self.rng.normal(0, self.cfg.gyro_std, 3)
        acc = raw[3:] + self.rng.normal(0, self.cfg.accel_std, 3)
        return np.concatenate([gyro, acc])


class DepthCamera:
    """Astra-like depth image via ray casting (headless, deterministic).

    Returns z-depth along the optical axis (body +x) in metres; 0 = no return
    (closer than min_range or beyond max_range), as the real sensor reports.
    """

    def __init__(self, cfg, rcfg, model, rng):
        self.cfg, self.rng = cfg, rng
        self.site = model.site("depth_cam").id
        self.pos = model.site_pos[self.site].copy()  # in robot body frame
        az = np.tan(np.radians(cfg.hfov_deg / 2)) * np.linspace(1, -1, cfg.width)   # +left
        el = np.tan(np.radians(cfg.vfov_deg / 2)) * np.linspace(1, -1, cfg.height)  # +up
        A, E = np.meshgrid(az, el)
        self.dirs = np.stack([np.ones_like(A), A, E], -1)  # body frame, x-component 1 (pinhole)
        unit = self.dirs / np.linalg.norm(self.dirs, axis=-1, keepdims=True)
        self._unit = unit.reshape(-1, 3)
        self._xcomp = unit[..., 0].ravel()
        n = cfg.width * cfg.height
        self._gid, self._dist = np.zeros(n, np.int32), np.zeros(n)
        self._group = np.array([i != ROBOT_GROUP for i in range(6)], np.uint8)

    def read(self, model, data):
        R = data.site_xmat[self.site].reshape(3, 3)
        vec = (self._unit @ R.T).ravel()
        mujoco.mj_multiRay(model, data, data.site_xpos[self.site], vec, self._group, 1, -1,
                           self._gid, self._dist, None, len(self._dist), self.cfg.max_range * 2)
        z = self._dist * self._xcomp
        z = z + self.rng.normal(0, 1, z.shape) * self.cfg.noise_k * z ** 2
        z[(self._gid < 0) | (z < self.cfg.min_range) | (z > self.cfg.max_range)] = 0.0
        return z.reshape(self.cfg.height, self.cfg.width).astype(np.float32)

    def to_scan(self, depth, z_min=0.05, z_max=0.45):
        """depthimage_to_laserscan-style: nearest obstacle point per column, (N, 2) xy in body frame.

        Floor/ceiling points are dropped by height. A column whose near-horizontal pixels
        return nothing is reported at min_range: in the closed arena (diagonal < max_range)
        a missing horizontal return can only mean 'too close'.
        """
        pts = self.pos + depth[..., None] * self.dirs  # body frame (dirs have x=1)
        h = pts[..., 2]
        obst = (depth > 0) & (h > z_min) & (h < z_max)
        r = np.where(obst, np.hypot(pts[..., 0], pts[..., 1]), np.inf)
        row = r.argmin(0)
        cols = np.arange(depth.shape[1])
        xy = pts[row, cols, :2]
        found = np.isfinite(r[row, cols])
        horizon = np.abs(self.dirs[:, 0, 2]) < np.tan(np.radians(5))
        blind = (depth[horizon] == 0).any(0) & ~found
        xy[blind] = self.pos[:2] + self.cfg.min_range * self.dirs[0, blind, :2]
        return xy[found | blind]


class Bumper:
    """Kobuki bumpers (left / centre / right over the front half), latched over the control period."""

    def __init__(self, cfg, rcfg, model, rng):
        self.base = model.geom("base").id
        self.robot = model.body("robot").id
        self.hit = np.zeros(3)

    def accumulate(self, data):
        for c in data.contact[:data.ncon]:
            if self.base in (c.geom1, c.geom2):
                p = data.xmat[self.robot].reshape(3, 3).T @ (c.pos - data.xpos[self.robot])
                a = np.degrees(np.arctan2(p[1], p[0]))
                if abs(a) <= 90:
                    self.hit[0 if a > 20 else 2 if a < -20 else 1] = 1

    def read(self, data):
        out, self.hit = self.hit, np.zeros(3)
        return out


class Odometry:
    """Kobuki-style dead reckoning: distance from encoders, heading from gyro when available."""

    def __init__(self, rcfg):
        self.r, self.L, self.tpr = rcfg.wheel_radius, rcfg.wheel_sep, rcfg.ticks_per_rev
        self.pose = np.zeros(3)
        self.prev = None

    def update(self, ticks, gyro_z, dt):
        if self.prev is None:
            self.prev = ticks
            return self.pose.copy()
        dl, dr = (ticks - self.prev) * 2 * np.pi * self.r / self.tpr
        self.prev = ticks
        ds = (dl + dr) / 2
        dth = gyro_z * dt if gyro_z is not None else (dr - dl) / self.L
        th = self.pose[2] + dth / 2  # midpoint integration
        self.pose += [ds * np.cos(th), ds * np.sin(th), dth]
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi
        return self.pose.copy()
