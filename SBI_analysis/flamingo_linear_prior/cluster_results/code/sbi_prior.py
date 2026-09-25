"""CPU/GPU NPE prior adapter with the SAME nonrectangular support.

Sampling uses exact IID rejection. log_prob includes an explicitly estimated
normalizing constant Z from audit.json, with its reported integration error.
This constant cancels in posterior density ratios and support rejection. Do
not use it for precision evidence estimation. MCMC support transformations
are not supplied here; use NPE rejection sampling with this support.
"""
import numpy as np
import torch
from torch.distributions import Distribution, constraints

from prior import JointPrior


class JointSupport(constraints.Constraint):
    event_dim = 1

    def __init__(self, prior):
        self.prior = prior

    def check(self, value):
        values = value.detach().cpu().numpy()
        result = self.prior.contains(values)
        return torch.as_tensor(result, device=value.device).reshape(value.shape[:-1])


class ExtendedPrior(Distribution):
    arg_constraints = {}
    has_rsample = False

    def __init__(self, config, normalization, device="cpu", seed=1729):
        self.joint = JointPrior(config)
        self.normalization = float(normalization)
        if not 0 < self.normalization < 1:
            raise ValueError("Supply the conditional mass Z from audit.json")
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self._support = JointSupport(self.joint)
        super().__init__(batch_shape=torch.Size(),event_shape=torch.Size([9]),validate_args=False)

    @property
    def support(self):
        return self._support

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape)
        count = int(np.prod(shape)) if shape else 1
        values = self.joint.sample(count,self.rng)
        return torch.as_tensor(values,dtype=torch.float32,device=self.device).reshape(shape+(9,))

    def log_prob(self, value):
        low, high = self.joint.low, self.joint.high
        constant = -np.log(self.normalization)
        result = torch.zeros(value.shape[:-1],device=value.device,dtype=value.dtype)
        for k in range(9):
            if k in self.joint.config["log_uniform_indices"]:
                constant -= np.log(np.log(high[k]/low[k]))
                result = result-torch.log(value[...,k].clamp_min(torch.finfo(value.dtype).tiny))
            else:
                constant -= np.log(high[k]-low[k])
        return torch.where(self.support.check(value),result+constant,-torch.inf)
