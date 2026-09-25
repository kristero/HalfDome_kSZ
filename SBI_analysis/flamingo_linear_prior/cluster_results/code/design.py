"""Prefix-stable linear-prior design with statistically compatible old points.

Old density on the common support is proportional to 1/(P0*xc). Thinning
with probability P0*xc/(P0_max*xc_max) changes it to the new constant joint
density. Apply thinning to the ENTIRE frozen old design, independently of
which simulations have completed; completion time must not select parameters.

Both source designs are randomized Sobol designs, not IID realizations. This
preserves the target density, without claiming digital-net balance after cuts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from prior import JointPrior, digest

THINNING_SALT = "halfdome-linear-prior-thinning-v1:"
OLD_SLOT_STRIDE = 8


def old_pool(root, prior):
    root = Path(root)
    old_manifest = json.loads((root / "inputs/previous_manifest.json").read_text())
    old_config = old_manifest["prior"]
    # Every scientific support setting must match; only the base measure,
    # random stream, version and explanatory text may change.
    descriptive = {"version", "log_uniform_indices", "seed", "distribution", "interpretation"}
    old_science = {k:v for k,v in old_config.items() if k not in descriptive}
    new_science = {k:v for k,v in prior.config.items() if k not in descriptive}
    if old_science != new_science or old_config["log_uniform_indices"] != [0, 1]:
        raise ValueError("Density-ratio reuse requires exactly the same broad support")
    if prior.config["log_uniform_indices"]:
        raise ValueError("All new base coordinates must be linear-uniform")
    path = root / "inputs/previous_theta.npy"
    assert digest(path) == old_manifest["design_sha256"]["theta.npy"]
    theta = np.load(path, allow_pickle=False)
    probability = theta[:, 0] * theta[:, 1] / (prior.high[0] * prior.high[1])
    random_uniform = np.array([int.from_bytes(hashlib.sha256(
        (THINNING_SALT + str(i)).encode()).digest()[:8], "big") / 2**64 for i in range(len(theta))])
    assert np.all((probability >= 0) & (probability <= 1))
    selected = np.flatnonzero(random_uniform < probability)
    assert prior.contains(theta[selected]).all()
    return theta, selected, probability, random_uniform


def build_design(root, count):
    root = Path(root)
    run = json.loads((root / "run_config.json").read_text())
    prior = JointPrior(json.loads((root / "code/prior.json").read_text()))
    old_theta, selected, probability, uniform = old_pool(root, prior)
    old_slots = np.arange(len(selected), dtype=np.int64) * OLD_SLOT_STRIDE
    selected = selected[old_slots < count]
    old_slots = old_slots[old_slots < count]
    fresh_slots = np.setdiff1d(np.arange(count, dtype=np.int64), old_slots)
    if len(fresh_slots):
        fresh_theta, fresh_proposal = prior.sobol(len(fresh_slots))
    else:
        fresh_theta, fresh_proposal = np.empty((0, 9)), np.empty(0, dtype=np.int64)
    theta = np.empty((count, 9))
    theta[old_slots] = old_theta[selected]
    theta[fresh_slots] = fresh_theta
    old_row = np.full(count, -1, dtype=np.int64)
    old_row[old_slots] = selected
    # Disjoint identities for hashing a stable training/validation split.
    proposal_id = np.empty(count, dtype=np.int64)
    proposal_id[old_slots] = selected
    proposal_id[fresh_slots] = 10**9 + fresh_proposal
    seed_ids = run["fresh_noise_row_offset"] + np.arange(count, dtype=np.int64)
    seed_ids[old_slots] = selected
    return dict(theta=theta, proposal_index=proposal_id, source_old_row=old_row,
                noise_seed_row_ids=seed_ids, old_selected_rows=selected,
                old_slots=old_slots, fresh_slots=fresh_slots,
                fresh_proposal_index=fresh_proposal,
                thinning_probabilities=probability, thinning_uniforms=uniform)
