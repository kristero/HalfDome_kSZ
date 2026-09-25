"""Optional PyTorch wrapper with the same guarded support as generation."""
import numpy as np
import torch
from torch.distributions import Distribution, constraints


class _GuardedSupport(constraints.Constraint):
    event_dim = 1

    def __init__(self, prior):
        self.prior = prior

    def check(self, value):
        accepted = self.prior.contains(value.detach().cpu().numpy())
        return torch.as_tensor(accepted, device=value.device, dtype=torch.bool)


class GuardedTorchPrior(Distribution):
    arg_constraints = {}
    has_rsample = False

    def __init__(self, prior, device="cpu", dtype=None):
        self.prior = prior
        self.dtype = dtype or torch.float32
        self.device = torch.device(device)
        self.low = torch.as_tensor(prior.low, dtype=self.dtype, device=self.device)
        self.high = torch.as_tensor(prior.high, dtype=self.dtype, device=self.device)
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([2]), validate_args=False)

    @property
    def support(self):
        return _GuardedSupport(self.prior)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape)
        remaining = int(np.prod(shape)) if shape else 1
        parts = []
        while remaining:
            count = min(16384, max(1024, remaining*2))
            candidates = self.low + torch.rand(count,2,device=self.device,dtype=self.dtype)*(self.high-self.low)
            accepted = candidates[self.support.check(candidates)][:remaining]
            parts.append(accepted)
            remaining -= len(accepted)
        result = torch.cat(parts) if parts else torch.empty((0,2),device=self.device,dtype=self.dtype)
        return result.reshape(shape + self.event_shape)

    def log_prob(self, value):
        value = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        inside = self.support.check(value)
        density = torch.full(inside.shape, -np.log(self.prior.area), device=self.device, dtype=self.dtype)
        return torch.where(inside, density, torch.full_like(density,-torch.inf))
