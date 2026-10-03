"""All tunables in one place. Override any field with a JSON file: {"world": {"n_obstacles": 12}}."""
import json
from dataclasses import dataclass, field, asdict


@dataclass
class RobotCfg:  # Kobuki spec values
    wheel_radius: float = 0.035
    wheel_sep: float = 0.230
    max_v: float = 0.7          # m/s
    max_w: float = 3.14         # rad/s
    acc_v: float = 1.0          # m/s^2, TurtleBot 2 velocity smoother (yocs) defaults
    acc_w: float = 2.0          # rad/s^2
    ticks_per_rev: float = 2578.33


@dataclass
class EncoderCfg:
    enabled: bool = True


@dataclass
class ImuCfg:
    enabled: bool = True
    gyro_std: float = 0.005     # rad/s
    gyro_bias_std: float = 0.0005  # rad/s residual after Kobuki startup calibration
    accel_std: float = 0.05     # m/s^2


@dataclass
class DepthCfg:  # Orbbec Astra-like
    enabled: bool = True
    width: int = 80             # downsampled from 640x480
    height: int = 60
    hfov_deg: float = 60.0
    vfov_deg: float = 49.5
    min_range: float = 0.6
    max_range: float = 8.0
    noise_k: float = 1.425e-3   # sigma = k*z^2 (structured light, Khoshelham & Elberink 2012)


@dataclass
class BumperCfg:
    enabled: bool = True


@dataclass
class SensorCfg:
    encoders: EncoderCfg = field(default_factory=EncoderCfg)
    imu: ImuCfg = field(default_factory=ImuCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    bumper: BumperCfg = field(default_factory=BumperCfg)


@dataclass
class WorldCfg:
    size: float = 5.6           # square arena side, m (diagonal < Astra max range)
    n_obstacles: int = 8
    obstacle_radius: tuple = (0.15, 0.35)  # box half-size / cylinder radius range
    box_fraction: float = 0.5   # share of obstacles that are boxes (rest cylinders)
    min_gap: float = 0.0        # m between obstacle bounding circles; 0 = no spacing rule
    min_start_goal: float = 3.5


@dataclass
class EpisodeCfg:
    control_hz: float = 20.0
    timestep: float = 0.002
    max_time: float = 60.0      # s sim time
    goal_tol: float = 0.25      # m


@dataclass
class Config:
    robot: RobotCfg = field(default_factory=RobotCfg)
    sensors: SensorCfg = field(default_factory=SensorCfg)
    world: WorldCfg = field(default_factory=WorldCfg)
    episode: EpisodeCfg = field(default_factory=EpisodeCfg)


def _merge(obj, d):
    for k, v in d.items():
        cur = getattr(obj, k)  # AttributeError on typos, on purpose
        if isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(obj, k, type(cur)(v) if not isinstance(cur, tuple) else tuple(v))


def load(path=None):
    cfg = Config()
    if path:
        with open(path) as f:
            _merge(cfg, json.load(f))
    return cfg


def to_dict(cfg):
    return asdict(cfg)
