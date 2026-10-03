import numpy as np


class Controller:
    """Interface: act(obs) -> (v [m/s], w [rad/s]). obs is NavEnv's observation dict;
    obs["features"] follows nav.features.names()."""

    def __init__(self, cfg, **params):
        self.cfg = cfg

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError


def split(f):
    """features -> goal_dist, bearing, front corridor clearance, left (30-90 deg) and
    right (-30..-90 deg) clearance, any bumper pressed. See nav.features.names()."""
    s = f[4:16]
    return f[0], np.arctan2(f[1], f[2]), f[3], s[1:4].min(), s[9:12].min(), f[16:19].any()
