"""Flat physical P0/beta prior with the other seven parameters fixed at B12.

The full rectangle is the support. Historical profile guards are diagnostics
only and must not be used to silently filter or redraw these samples.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent


class FlatPrior:
    def __init__(self, config=None):
        self.config = config or json.loads((HERE / "prior.json").read_text())
        self.names = self.config["parameter_order"]
        if self.names != ["P0", "beta"]:
            raise ValueError("Expected physical parameter order [P0, beta]")
        if self.config["rejection_cuts"] or self.config["log_uniform_indices"]:
            raise ValueError("This prior requires linear sampling without cuts")
        self.low = np.asarray(self.config["lower"], dtype=float)
        self.high = np.asarray(self.config["upper"], dtype=float)
        if not np.all(np.isfinite([self.low, self.high])) or not np.all(self.low < self.high):
            raise ValueError("Bounds must be finite and strictly increasing")
        self.width = self.high - self.low
        self.area = float(np.prod(self.width))

    @staticmethod
    def _parameters(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.ndim == 0 or theta.shape[-1] != 2:
            raise ValueError("Expected a final dimension of two: P0, beta")
        return theta

    def contains(self, theta):
        theta = self._parameters(theta)
        return np.all(np.isfinite(theta) & (theta >= self.low) & (theta <= self.high), axis=-1)

    def log_prob(self, theta):
        """Normalized density in physical coordinates, including outside support."""
        return np.where(self.contains(theta), -np.log(self.area), -np.inf)

    def sample(self, count, seed=None):
        """Independent random samples for inference; no rejection or clipping."""
        if count < 0 or int(count) != count:
            raise ValueError("count must be a nonnegative integer")
        return np.random.default_rng(seed).uniform(self.low, self.high, size=(int(count), 2))

    def sobol(self, count, seed=None):
        """Prefix-stable space-filling design; power-of-two counts are preferred."""
        if count < 0 or int(count) != count:
            raise ValueError("count must be a nonnegative integer")
        count = int(count)
        if count == 0:
            return np.empty((0, 2))
        seed = self.config["design_seed"] if seed is None else seed
        unit = qmc.Sobol(2, scramble=True, seed=seed).random_base2((count - 1).bit_length())[:count]
        return self.low + unit * self.width

    def expand(self, theta):
        """Insert the seven user-confirmed fixed B12 values in original order."""
        theta = self._parameters(theta)
        order = self.config["full_parameter_order"]
        full = np.empty(theta.shape[:-1] + (len(order),), dtype=float)
        for index, name in enumerate(order):
            if name in self.names:
                full[..., index] = theta[..., self.names.index(name)]
            else:
                full[..., index] = self.config["fixed_parameters"][name]
        return full

    def as_torch_distribution(self, device="cpu"):
        """Optional normalized two-dimensional distribution for SBI workflows."""
        import torch
        low = torch.tensor(self.low, dtype=torch.float32, device=device)
        high = torch.tensor(self.high, dtype=torch.float32, device=device)
        return torch.distributions.Independent(
            torch.distributions.Uniform(low, high, validate_args=False), 1)
