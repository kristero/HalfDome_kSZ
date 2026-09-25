"""Construct the exact normalized box prior for future NPE/SBI training.

The target is independent linear-uniform physical coordinates. There is no
estimated rejection normalization and no Gaussian or log-uniform component.
"""
import torch
from torch.distributions import Independent, Uniform

from prior import UniformPrior


def make_prior(config=None, device="cpu"):
    box = UniformPrior(config)
    box.certify_box()
    low = torch.as_tensor(box.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(box.high, dtype=torch.float32, device=device)
    return Independent(Uniform(low, high, validate_args=False), 1)
