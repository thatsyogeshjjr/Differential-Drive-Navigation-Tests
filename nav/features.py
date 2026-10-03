"""Perception -> fixed-size feature vector shared by all controllers (and stored in the dataset).

The Astra sees only 60 deg, so recent scan points are kept for a few seconds in the odometry
frame (a tiny rolling costmap). Features are then computed over all 360 deg around the robot.
"""
from collections import deque

import numpy as np

from .world import ROBOT_RADIUS as R

N_SECTORS = 12        # 30 deg each, sector 0 centred straight ahead, counter-clockwise
HORIZON = 3.0         # m, ignore points further than this
MEMORY_S = 3.0        # s of scan history (odometry drift over 3 s is ~mm)


def names():
    return (["goal_dist", "goal_sin", "goal_cos", "front_corridor"]
            + [f"clear{i * 30}" for i in range(N_SECTORS)] + ["bump_l", "bump_c", "bump_r"])


class LocalMap:
    def __init__(self, control_hz):
        self.buf = deque(maxlen=int(MEMORY_S * control_hz))

    def update(self, odom, pts):
        c, s = np.cos(odom[2]), np.sin(odom[2])
        self.buf.append(odom[:2] + pts @ np.array([[c, s], [-s, c]]))

    def body_points(self, odom):
        P = np.concatenate(self.buf) - odom[:2] if self.buf else np.zeros((0, 2))
        c, s = np.cos(odom[2]), np.sin(odom[2])
        return P @ np.array([[c, -s], [s, c]])

    def features(self, odom, goal, bumper):
        P = self.body_points(odom)
        r = np.hypot(P[:, 0], P[:, 1])
        P, r = P[r < HORIZON], r[r < HORIZON]
        a = np.arctan2(P[:, 1], P[:, 0])
        sectors = np.full(N_SECTORS, HORIZON - R)
        np.minimum.at(sectors, ((a + np.pi / N_SECTORS) % (2 * np.pi) // (2 * np.pi / N_SECTORS)).astype(int), r - R)
        ahead = (P[:, 0] > 0) & (np.abs(P[:, 1]) < R + 0.05)  # swept corridor of the robot's width
        front = P[ahead, 0].min() - R if ahead.any() else HORIZON - R
        dx, dy = goal[0] - odom[0], goal[1] - odom[1]
        b = np.arctan2(dy, dx) - odom[2]
        return np.array([np.hypot(dx, dy), np.sin(b), np.cos(b), front, *sectors, *bumper], np.float32)
