"""NavEnv: one seeded episode = one generated arena. step(v, w) runs one control period."""
import mujoco
import numpy as np

from . import features, world
from .sensors import Bumper, DepthCamera, Encoders, Imu, Odometry


def _quat(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


class NavEnv:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, seed):
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        self.world = world.generate(cfg.world, rng)
        self.model = mujoco.MjModel.from_xml_string(
            world.to_mjcf(self.world, cfg.episode.timestep), {"kobuki.xml": world.ROBOT_XML})
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        x, y, yaw = self.world["start"]
        d.qpos[:7] = [x, y, 0.001, *_quat(yaw)]
        mujoco.mj_forward(m, d)
        for _ in range(int(0.3 / m.opt.timestep)):  # settle onto wheels/casters
            mujoco.mj_step(m, d)
        d.time = 0.0

        srng = np.random.default_rng([seed, 1])  # sensor noise stream, independent of world
        s, r = cfg.sensors, cfg.robot
        self.enc = Encoders(s.encoders, r, m, srng) if s.encoders.enabled else None
        self.imu = Imu(s.imu, r, m, srng) if s.imu.enabled else None
        self.cam = DepthCamera(s.depth, r, m, srng) if s.depth.enabled else None
        self.bumper = Bumper(s.bumper, r, m, srng) if s.bumper.enabled else None
        assert self.enc, "odometry needs wheel encoders"
        self.odom = Odometry(r)
        self.local_map = features.LocalMap(cfg.episode.control_hz)

        robot = m.body("robot").id
        self._robot_geoms = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == robot}
        self._obstacle_geoms = {g for g in range(m.ngeom) if m.geom(g).name.startswith(("obs", "wall"))}
        # goal expressed in the odometry frame (= start pose), like a move_base goal
        c, s_ = np.cos(yaw), np.sin(yaw)
        gx, gy = self.world["goal"][0] - x, self.world["goal"][1] - y
        self.goal_odom = np.array([c * gx + s_ * gy, -s_ * gx + c * gy])
        self.n_substeps = round(1 / (cfg.episode.control_hz * m.opt.timestep))
        self.path_length, self.collisions, self._last_contact = 0.0, 0, -np.inf
        self.cmd = np.zeros(2)
        return self._observe()

    def gt_pose(self):
        q = self.data.qpos
        return np.array([q[0], q[1], np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))])

    def _touching(self):
        d = self.data
        for c in d.contact[:d.ncon]:
            a, b = c.geom1, c.geom2
            if (a in self._robot_geoms and b in self._obstacle_geoms) or (b in self._robot_geoms and a in self._obstacle_geoms):
                return True
        return False

    def step(self, v, w):
        r, dt = self.cfg.robot, 1 / self.cfg.episode.control_hz
        target = np.clip([v, w], [-r.max_v, -r.max_w], [r.max_v, r.max_w])
        self.cmd += np.clip(target - self.cmd, -np.array([r.acc_v, r.acc_w]) * dt, np.array([r.acc_v, r.acc_w]) * dt)
        v, w = self.cmd
        self.data.ctrl[:] = [(v - w * r.wheel_sep / 2) / r.wheel_radius, (v + w * r.wheel_sep / 2) / r.wheel_radius]
        for _ in range(self.n_substeps):
            p0 = self.data.qpos[:2].copy()
            mujoco.mj_step(self.model, self.data)
            for sensor in (self.imu, self.bumper):
                if sensor:
                    sensor.accumulate(self.data)
            self.path_length += np.hypot(*(self.data.qpos[:2] - p0))
            if self._touching():  # one collision event = contact separated by >= 0.5 s of no contact
                self.collisions += self.data.time - self._last_contact > 0.5
                self._last_contact = self.data.time
        obs = self._observe()
        gt = self.gt_pose()
        dist = np.hypot(*(gt[:2] - self.world["goal"]))
        # success is judged on ground truth; the robot "declaring" arrival (odometry within
        # 5 cm of the goal) also ends the episode, so odometry drift shows up as failures
        believed = obs["features"][0] < 0.05
        outcome = ("success" if dist < self.cfg.episode.goal_tol else "odom_arrived_off_goal" if believed
                   else "timeout" if self.data.time >= self.cfg.episode.max_time else None)
        info = dict(gt_pose=gt, goal_dist=dist, success=outcome == "success", outcome=outcome,
                    collisions=self.collisions, path_length=self.path_length, cmd=(v, w))
        return obs, outcome is not None, info

    def _observe(self):
        d, dt = self.data, 1 / self.cfg.episode.control_hz
        ticks = self.enc.read(d)
        imu = self.imu.read(d) if self.imu else None
        odom = self.odom.update(ticks, imu[2] if imu is not None else None, dt)
        depth = self.cam.read(self.model, d) if self.cam else None
        bump = self.bumper.read(d) if self.bumper else np.zeros(3)
        if self.cam:
            self.local_map.update(odom, self.cam.to_scan(depth))
        return dict(t=d.time, ticks=ticks, imu=imu, depth=depth, bumper=bump, odom=odom, goal=self.goal_odom,
                    features=self.local_map.features(odom, self.goal_odom, bump))
