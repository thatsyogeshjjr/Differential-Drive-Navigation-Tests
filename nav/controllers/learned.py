"""Supervised controller: MLP trained by `python -m nav train` on recorded demonstrations."""
import numpy as np

from . import register
from .base import Controller
from .. import features
from ..train import forward


@register("learned")
class Learned(Controller):
    def __init__(self, cfg, model="models/bc.npz", **_):
        super().__init__(cfg)
        z = np.load(model)
        assert list(z["feature_names"]) == features.names(), "model trained on a different feature set"
        self.xm, self.xs, self.ym, self.ys = z["xm"], z["xs"], z["ym"], z["ys"]
        self.params = [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]

    def act(self, obs):
        f = obs["features"]
        y = forward(self.params, (f - self.xm) / self.xs)[-1] * self.ys + self.ym
        return min(float(y[0]), 0.5 * f[0] + 0.05), float(y[1])
