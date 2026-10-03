"""Rule-based: go-to-goal, and when the corridor ahead is blocked, turn and follow the
obstacle boundary until the goal direction is free again (a Bug-algorithm flavour).
Bumper hit -> back off first."""
import numpy as np

from . import register
from .base import Controller, split

GO, TURN, FOLLOW, BACK = range(4)


@register("rule")
class Rule(Controller):
    def __init__(self, cfg, v_max=0.4, stop=0.3, wall=0.3, **_):
        super().__init__(cfg)
        self.v_max, self.stop, self.wall = float(v_max), float(stop), float(wall)
        self.hz = cfg.episode.control_hz
        self.reset()

    def reset(self):
        self.mode, self.dir, self.timer = GO, 1, 0

    def _goal_free(self, f, dist, bearing):
        k = int(round(bearing / (np.pi / 6))) % 12
        return f[4 + k] > min(dist, 1.0) and f[4 + (k - 1) % 12] > 0.3 and f[4 + (k + 1) % 12] > 0.3

    def act(self, obs):
        f = obs["features"]
        dist, bearing, front, left, right, bump = split(f)
        self.timer += 1
        if bump and self.mode != BACK:
            self.mode, self.timer = BACK, 0
            self.dir = -1 if f[16] else 1          # bumped on the left -> turn right afterwards
        if self.mode == BACK:
            if self.timer < 0.6 * self.hz:
                return -0.15, 0.0
            self.mode, self.timer = TURN, 0
        if self.mode == GO and front < self.stop:
            self.mode, self.timer = TURN, 0
            self.dir = 1 if left >= right else -1
        if self.mode == TURN:
            if front > self.stop + 0.3:
                self.mode, self.timer = FOLLOW, 0
            else:
                return 0.0, 1.0 * self.dir
        if self.mode == FOLLOW:
            if front < self.stop:
                self.mode, self.timer = TURN, 0
                return 0.0, 1.0 * self.dir
            if (self.timer > self.hz and self._goal_free(f, dist, bearing)) or self.timer > 10 * self.hz:
                self.mode = GO
            else:  # keep the obstacle on the side opposite to the turn, at `wall` clearance
                side = f[4 + 8:4 + 11].min() if self.dir > 0 else f[4 + 2:4 + 5].min()
                w = -self.dir * np.clip(2.0 * (side - self.wall), -1.0, 1.0)
                return self.v_max * np.clip(front / 0.8, 0.3, 1.0), w
        side = max(0.0, 0.25 - left) - max(0.0, 0.25 - right)  # nudge away from close sides
        w = 1.5 * bearing - 2.0 * side
        v = self.v_max * np.clip((front - self.stop) / 0.8, 0.25, 1.0) * max(0.0, np.cos(bearing)) ** 2
        return min(v, 0.5 * dist + 0.05), w
