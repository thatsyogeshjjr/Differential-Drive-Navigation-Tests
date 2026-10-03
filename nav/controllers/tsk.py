"""First-order Takagi-Sugeno-Kang fuzzy controller.

Inputs  z = [1, bearing, front, left, right]  (rad, m clearance)
Rule i: IF antecedents THEN v_i = a_i . z, w_i = c_i . z
Output: weighted average of rule outputs, weights = product of Gaussian memberships.
Edge sets are shoulders (membership 1 beyond the centre).
"""
import numpy as np

from . import register
from .base import Controller, split

# (centre, sigma, shoulder) ; shoulder -1 = left shoulder, +1 = right shoulder, 0 = plain Gaussian
MF = {
    "b": {"neg": (-1.2, 0.5, -1), "zero": (0.0, 0.45, 0), "pos": (1.2, 0.5, 1)},
    "F": {"near": (0.25, 0.15, -1), "far": (0.9, 0.3, 1)},
    "L": {"near": (0.15, 0.12, -1), "far": (0.5, 0.15, 1)},
    "R": {"near": (0.15, 0.12, -1), "far": (0.5, 0.15, 1)},
}

#          antecedents                       v coeffs [1, b, F, L, R]     w coeffs [1, b, F, L, R]
RULES = [
    ({"F": "far", "b": "zero"},              [0.40, 0, 0, 0, 0],          [0, 1.5, 0, 0, 0]),     # cruise to goal
    ({"F": "far", "b": "neg"},               [0.05, 0, 0, 0, 0],          [0, 1.2, 0, 0, 0]),     # goal right: turn
    ({"F": "far", "b": "pos"},               [0.05, 0, 0, 0, 0],          [0, 1.2, 0, 0, 0]),     # goal left: turn
    ({"F": "near"},                          [0.0, 0, 0, 0, 0],           [0.4, 0, 0, 2.5, -2.5]),  # blocked: turn to open side
    ({"F": "near", "L": "near", "R": "near"}, [-0.08, 0, 0, 0, 0],        [1.0, 0, 0, 0, 0]),     # boxed in: back + spin
    ({"L": "near", "b": "pos"},              [0.25, 0, 0, 0, 0],          [-0.5, 0, 0, 0, 0]),    # obstacle left, goal left: skirt it
    ({"R": "near", "b": "neg"},              [0.25, 0, 0, 0, 0],          [0.5, 0, 0, 0, 0]),     # obstacle right, goal right: skirt it
]


def mu(x, c, s, shoulder):
    if shoulder * (x - c) > 0:
        return 1.0
    return float(np.exp(-0.5 * ((x - c) / s) ** 2))


@register("tsk")
class TSK(Controller):
    def act(self, obs):
        dist, bearing, front, left, right, _ = split(obs["features"])
        x = {"b": bearing, "F": front, "L": left, "R": right}
        z = np.array([1.0, bearing, front, left, right])
        w = np.array([np.prod([mu(x[k], *MF[k][lab]) for k, lab in ante.items()]) for ante, _, _ in RULES])
        V = np.array([a for _, a, _ in RULES]) @ z
        W = np.array([c for _, _, c in RULES]) @ z
        s = w.sum() + 1e-9
        v, om = w @ V / s, w @ W / s
        return min(v, 0.5 * dist + 0.05), om
