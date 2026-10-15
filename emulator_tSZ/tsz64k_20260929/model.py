"""Noiseless 40-band tSZ emulator with exact pressure-amplitude scaling.

The network learns log10(D_ell / P0**2) from the other eight parameters.
Multiplication by P0**2 is exact for the fixed-catalogue pressure painter,
including its linear beam and mask operations.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

PARAMETERS = (
    "P0", "xc", "beta", "alpha_m_P0", "alpha_m_xc", "alpha_m_beta",
    "alpha_z_P0", "alpha_z_xc", "alpha_z_beta",
)


def features(theta):
    """Use logarithmic positive shape scales and linear evolution exponents."""
    values = np.asarray(theta, dtype=np.float64)[:, 1:].copy()
    values[:, :2] = np.log10(values[:, :2])
    return values


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                    nn.Linear(width, width))

    def forward(self, x):
        return x + self.layers(self.norm(x)) / np.sqrt(2.0)


class SpectrumNetwork(nn.Module):
    def __init__(self, width=256, blocks=4):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(8, width), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 40))

    def forward(self, x):
        return self.output(self.blocks(self.input(x)))


class Emulator:
    """Load a selected model or log-space ensemble; return linear D_ell."""
    def __init__(self, artifact, device="cpu"):
        self.device = torch.device(device)
        self.artifact = torch.load(Path(artifact), map_location="cpu", weights_only=False)
        self.models = []
        for entry in self.artifact["members"]:
            model = SpectrumNetwork(**entry["config"])
            model.load_state_dict(entry["state_dict"])
            self.models.append(model.to(self.device).eval())

    def predict(self, theta, batch_size=2048):
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if theta.shape[1] != 9 or not np.isfinite(theta).all():
            raise ValueError("Expected a finite (N,9) array in the saved parameter order")
        a = self.artifact
        if np.any(theta < a["lower"]) or np.any(theta > a["upper"]):
            raise ValueError("Parameters lie outside the 64k training prior; extrapolation is disabled")
        x = (features(theta) - a["feature_mean"]) / a["feature_std"]
        result = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                batch = torch.as_tensor(x[start:start+batch_size], dtype=torch.float32,
                                        device=self.device)
                scaled = torch.stack([model(batch) for model in self.models]).mean(0)
                log_shape = scaled.cpu().numpy().astype(float) * a["target_std"] + a["target_mean"]
                log_dl = log_shape + 2 * np.log10(theta[start:start+batch_size, 0, None])
                result.append(10.0 ** log_dl)
        prediction = np.concatenate(result)
        if not np.isfinite(prediction).all() or np.any(prediction <= 0):
            raise FloatingPointError("Nonfinite or nonpositive emulator prediction")
        return prediction
