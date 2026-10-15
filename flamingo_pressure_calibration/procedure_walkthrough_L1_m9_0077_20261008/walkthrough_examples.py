"""Read, rebuild and refit one existing FLAMINGO stack without particle I/O.

Run with the HalfDome science Python. This script prints JSON and never writes
into the completed study. All particle inputs in measurements.npz have already
been converted using the snapshot's own physical-CGS attributes.

Examples:
    python -B walkthrough_examples.py --bin-low 14.5 --radius 4
    python -B walkthrough_examples.py --bin-low 14.5 --radius 3 --field electrons

These examples reproduce the empirical, diagonal-weight fits, not a posterior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


DEFAULT_STUDY = (
    Path(__file__).resolve().parent.parent
    / "radii234_contamination_L1_m9_0077_20261008"
)


def load_code(study: Path):
    """Use the actual study's equations and optimizer, without running main()."""
    sys.path.insert(0, str(study))
    import pilot_physics as physics
    from fit_study import accepted, fit_one, resolution_masks
    from models import shell

    return physics, accepted, fit_one, resolution_masks, shell


def physical_shell_profiles(sums, halos, baryon_fraction, physics):
    """Rebuild all-gas normalized profiles from additive shell sums.

    sums: [halo, component, statistic, shell]. Components are exclusive.
    Y has physical Mpc^2 units; electron sums are dimensionless particle counts.
    Shell volume is geometric, not the sum of SPH particle volumes.
    """
    total = sums.sum(axis=1)
    index = {name: i for i, name in enumerate(physics.SUM_NAMES)}
    radius = halos["r_200_crit_mpc"]
    mass = halos["m200c_msun"]
    volume_mpc3 = (4 * np.pi / 3) * radius[:, None] ** 3 * np.diff(physics.EDGES**3)
    rho_critical = mass / ((4 * np.pi / 3) * radius**3) / 200
    pressure_200 = physics.p200(mass, radius, baryon_fraction)

    # Convert sum(Y) / V from Mpc^-1 to m^-1 before multiplying by SI constants.
    pressure_pa = (
        total[:, index["y"]] * physics.electron_mass * physics.c**2
        / (physics.SIGMA_T * volume_mpc3 * physics.MPC_M)
    )
    electrons_mpc3 = total[:, index["electrons"]] / volume_mpc3
    return {
        "pressure": pressure_pa / pressure_200[:, None],
        "electrons": (
            electrons_mpc3 * physics.MU_XGP_MSUN
            / (baryon_fraction * rho_critical[:, None])
        ),
    }


def choose_bin(halos, bin_low):
    """Use the stored bin assignment; do not silently reselect the population."""
    matching = np.flatnonzero(np.isclose(halos["mass_edges"][:-1], bin_low))
    if len(matching) != 1:
        raise ValueError("--bin-low must match a stored edge, e.g. 14.5")
    number = int(matching[0])
    rows = np.flatnonzero(halos["mass_bin"] == number)
    low, high = halos["mass_edges"][number : number + 2]
    label = f"bin_{low:.1f}_{high:.1f}".replace(".", "p")
    return rows, label


def joint_parameter_draw(parameter_file, radius_index, seed):
    """Draw one observed six-parameter row, preserving measured correlations.

    This is a demonstration, not a certified stochastic painting prescription.
    Only halos with stable, resolved fits in both fields enter this distribution.
    The resulting selection can bias the apparent scatter.
    """
    with np.load(parameter_file) as saved:
        good = saved["accepted"][:, radius_index].all(axis=1)
        parameters = saved["parameters"][good, radius_index].reshape(-1, 6)
        halo_ids = saved["halo_ids"][good]
    if not len(parameters):
        return {"accepted_joint_halos": 0, "draw": None}
    index = int(np.random.default_rng(seed).integers(len(parameters)))
    return {
        "accepted_joint_halos": len(parameters),
        "halo_id": int(halo_ids[index]),
        "parameter_order": ["P0", "xc", "beta_P", "rho0", "alpha", "beta_e"],
        "draw": parameters[index].tolist(),
        "limitation": "conditional resolved/stable halo distribution; not intrinsic scatter",
    }


def inspect_and_refit(study, bin_low=14.5, radius=4, field="pressure", seed=42):
    """Check units/stacking and reproduce one fit from the existing accumulators."""
    physics, accepted, fit_one, resolution_masks, shell = load_code(study)
    plan = json.loads((study / "metadata_plan.json").read_text())
    with np.load(study / "selected_halos.npz") as saved:
        halos = dict(saved)
    rows, label = choose_bin(halos, bin_low)
    key = {"pressure": "pressure_components", "electrons": "electron_components"}[field]
    with np.load(study / "measurements.npz") as saved:
        data = {name: saved[name] for name in ("sums", "hsml_median_upper_x")}
        values = saved[key][rows].sum(axis=1)
        small_halos = {k: halos[k][rows] for k in ("r_200_crit_mpc", "m200c_msun")}
        rebuilt = physical_shell_profiles(
            data["sums"][rows], small_halos, plan["baryon_fraction"], physics
        )[field]
        # Allow tiny changes in physical constants between science environments.
        # The separate optimizer check below uses the stored profiles directly.
        np.testing.assert_allclose(rebuilt, values, rtol=2e-8, atol=1e-25)
        rebuild_relative_error = float(np.max(
            np.abs(rebuilt - values) / np.maximum(np.abs(values), 1e-100)
        ))

    # Normalize each halo first; values now have [halo, shell] shape.
    mean = values.mean(axis=0)
    scatter = values.std(axis=0, ddof=1)
    base, _, _, supported = resolution_masks(
        halos, data, rows, plan["softening_physical_mpc"]
    )
    mask = base & (physics.EDGES[1:] <= radius + 1e-12) & (mean > 0)
    family = "b12" if field == "pressure" else "b16"
    fit = fit_one(family, mean, scatter, mask)
    if fit is None:
        raise RuntimeError("Fewer than six usable shells in this selection")
    stored = json.loads((study / "products" / f"{label}_fits.json").read_text())
    radius_index = (2, 3, 4).index(radius)
    reference = stored["selections"]["all"][field][radius_index]["fit"]
    np.testing.assert_allclose(fit["parameters"], reference["parameters"], rtol=1e-11)
    prediction = shell(family, fit["parameters"])
    rms = 100 * np.sqrt(np.mean((prediction[mask] / mean[mask] - 1) ** 2))
    np.testing.assert_allclose(rms, reference["rms_percent"], rtol=1e-11)
    with np.load(study / "products" / f"{label}_profiles.npz") as saved:
        np.testing.assert_allclose(mean, saved[f"all_{field}_mean"], rtol=1e-13)

    swift = json.loads((study / "swift_checks.json").read_text())
    return {
        "study": str(study),
        "bin": label,
        "halos": len(rows),
        "field": field,
        "radius_R200c": radius,
        "fitted_shells": int(mask.sum()),
        "strict_resolution_supported": bool(supported),
        "fit": fit,
        "optimizer_stable": accepted(fit),
        "rebuild_stack_and_refit_checks_passed": True,
        "rebuilt_profile_max_relative_difference": rebuild_relative_error,
        "SWIFT_checked_halos": swift["checked_halos"],
        "SWIFT_particle_ID_hashes_match": swift["all_particle_hashes_match"],
        "joint_draw_example": joint_parameter_draw(
            study / "products" / f"{label}_parameters.npz", radius_index, seed
        ),
        "snapshot_particles_read": False,
        "files_written": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--bin-low", type=float, default=14.5)
    parser.add_argument("--radius", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--field", choices=("pressure", "electrons"), default="pressure")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = inspect_and_refit(
        args.study.resolve(), args.bin_low, args.radius, args.field, args.seed
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
