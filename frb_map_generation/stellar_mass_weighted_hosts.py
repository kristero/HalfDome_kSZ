"""
Stellar-mass-weighted FRB host selection helpers.

This file is intentionally independent of the FRB DM integration code. Use it
to choose source positions, then compute foreground DM with the existing
foreground-map / DM-integration workflow.

Required inputs are existing NumPy arrays, for example:
    z_halo
    Mstar
    ra_halo, dec_halo

Only NumPy and Matplotlib are used here.
"""

import numpy as np
import matplotlib.pyplot as plt


def stellar_mass_weights(Mstar, alpha_star=1.0, eps=1e-30):
    """
    Return normalized probabilities proportional to Mstar**alpha_star.

    Parameters
    ----------
    Mstar : array_like
        Stellar masses in Msun.
    alpha_star : float
        Stellar-mass weighting exponent.
    eps : float
        Values <= eps are treated as zero weight.

    Returns
    -------
    p : ndarray
        Normalized probability array with sum(p) == 1.
    """
    Mstar = np.asarray(Mstar, dtype=float)

    if Mstar.ndim != 1:
        raise ValueError("Mstar must be a one-dimensional array.")
    if not np.isfinite(alpha_star):
        raise ValueError("alpha_star must be finite.")
    if alpha_star < 0:
        raise ValueError("alpha_star must be >= 0.")
    if eps < 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and non-negative.")

    valid = np.isfinite(Mstar) & (Mstar > eps)
    weights = np.zeros_like(Mstar, dtype=float)

    if np.any(valid):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            weights[valid] = Mstar[valid] ** alpha_star

    weights[~np.isfinite(weights)] = 0.0
    total = weights.sum(dtype=float)

    if total <= 0.0:
        raise ValueError(
            "All stellar-mass weights are zero. Check Mstar values, alpha_star, and eps."
        )

    return weights / total


def fixed_redshift_shell_mask(z_halo, z_source, dz=0.02):
    """
    Boolean mask for halos inside abs(z_halo - z_source) < dz / 2.
    """
    z_halo = np.asarray(z_halo, dtype=float)

    if z_halo.ndim != 1:
        raise ValueError("z_halo must be a one-dimensional array.")
    if not np.isfinite(z_source):
        raise ValueError("z_source must be finite.")
    if dz <= 0.0 or not np.isfinite(dz):
        raise ValueError("dz must be positive and finite.")

    return np.isfinite(z_halo) & (np.abs(z_halo - z_source) < 0.5 * dz)


def print_stellar_mass_weight_diagnostics(Mstar_shell, p_shell):
    """
    Print shell and weighting diagnostics.
    """
    Mstar_shell = np.asarray(Mstar_shell, dtype=float)
    p_shell = np.asarray(p_shell, dtype=float)

    positive = np.isfinite(Mstar_shell) & (Mstar_shell > 0.0)
    if not np.any(positive):
        raise ValueError("No positive finite stellar masses in shell.")

    positive_mstar = Mstar_shell[positive]
    n_eff = 1.0 / np.sum(p_shell**2)
    p_sorted = np.sort(p_shell)[::-1]

    print(f"number of halos in shell: {Mstar_shell.size}")
    print(f"positive finite Mstar in shell: {positive_mstar.size}")
    print(
        "Mstar shell min/median/max [Msun]: "
        f"{np.min(positive_mstar):.6e}, "
        f"{np.median(positive_mstar):.6e}, "
        f"{np.max(positive_mstar):.6e}"
    )
    print(f"N_eff = {n_eff:.6f}")

    for frac in (0.01, 0.10, 0.50):
        top_n = max(1, int(np.ceil(frac * p_sorted.size)))
        contribution = np.sum(p_sorted[:top_n])
        print(
            f"top {100 * frac:.0f}% halos: "
            f"{top_n} halos contribute {100 * contribution:.3f}% of probability"
        )


def sample_stellar_mass_weighted_hosts(
    z_halo,
    Mstar,
    n_frb,
    z_source,
    dz=0.02,
    alpha_star=1.0,
    seed=42,
):
    """
    Sample FRB host halo indices from a fixed redshift shell.

    Selection probability inside the shell is:
        p_i = Mstar_i**alpha_star / sum_j Mstar_j**alpha_star

    Parameters
    ----------
    z_halo : array_like
        Halo redshifts.
    Mstar : array_like
        Halo stellar masses in Msun.
    n_frb : int
        Number of FRB host halos to sample. Sampling is with replacement.
    z_source : float
        Source redshift at the center of the shell.
    dz : float
        Shell width. The cut is abs(z_halo - z_source) < dz / 2.
    alpha_star : float
        Stellar-mass weighting exponent.
    seed : int
        Random seed for np.random.default_rng.

    Returns
    -------
    host_indices : ndarray
        Original halo indices selected as FRB hosts.
    p_shell : ndarray
        Probability array for halos inside the shell, in shell-index order.
    """
    z_halo = np.asarray(z_halo, dtype=float)
    Mstar = np.asarray(Mstar, dtype=float)

    if z_halo.ndim != 1 or Mstar.ndim != 1:
        raise ValueError("z_halo and Mstar must be one-dimensional arrays.")
    if z_halo.shape != Mstar.shape:
        raise ValueError("z_halo and Mstar must have the same shape.")
    if int(n_frb) <= 0:
        raise ValueError("n_frb must be positive.")

    shell_mask = fixed_redshift_shell_mask(z_halo, z_source, dz=dz)
    shell_indices = np.flatnonzero(shell_mask)
    if shell_indices.size == 0:
        raise ValueError(
            f"No halos found in source shell abs(z_halo - {z_source}) < {0.5 * dz}."
        )

    Mstar_shell = Mstar[shell_indices]
    p_shell = stellar_mass_weights(Mstar_shell, alpha_star=alpha_star)
    print_stellar_mass_weight_diagnostics(Mstar_shell, p_shell)

    rng = np.random.default_rng(seed)
    local_choices = rng.choice(
        shell_indices.size,
        size=int(n_frb),
        replace=True,
        p=p_shell,
    )
    host_indices = shell_indices[local_choices]

    return host_indices, p_shell


def xyz_to_radec(x, y, z):
    """
    Convert Cartesian halo positions to RA/Dec in radians.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, and z must have the same shape.")

    r = np.sqrt(x * x + y * y + z * z)
    valid = np.isfinite(r) & (r > 0.0)
    if not np.all(valid):
        raise ValueError("All positions must have positive finite radius.")

    ra = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    dec = np.arcsin(z / r)
    return ra, dec


def plot_stellar_mass_host_histogram(
    Mstar_shell,
    Mstar_selected,
    bins=50,
    save_path="stellar_mass_weighted_host_histogram.png",
):
    """
    Plot all shell halo stellar masses and selected FRB host stellar masses.

    The x-axis is log10(Mstar/Msun). The y-axis is logarithmic.
    """
    Mstar_shell = np.asarray(Mstar_shell, dtype=float)
    Mstar_selected = np.asarray(Mstar_selected, dtype=float)

    valid_shell = np.isfinite(Mstar_shell) & (Mstar_shell > 0.0)
    valid_selected = np.isfinite(Mstar_selected) & (Mstar_selected > 0.0)

    if not np.any(valid_shell):
        raise ValueError("No positive finite shell stellar masses to plot.")
    if not np.any(valid_selected):
        raise ValueError("No positive finite selected host stellar masses to plot.")

    log_shell = np.log10(Mstar_shell[valid_shell])
    log_selected = np.log10(Mstar_selected[valid_selected])

    if isinstance(bins, int):
        lo = min(np.min(log_shell), np.min(log_selected))
        hi = max(np.max(log_shell), np.max(log_selected))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        bins = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(
        log_shell,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="All shell halos",
    )
    ax.hist(
        log_selected,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Selected FRB hosts",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax.set_ylabel("PDF")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


def example_usage(
    z_halo,
    Mstar,
    ra_halo,
    dec_halo,
    z_source=1.0,
    n_frb=10000,
    dz=0.02,
    alpha_star=1.0,
    seed=42,
    histogram_path="stellar_mass_weighted_host_histogram.png",
):
    """
    Small usage example wrapped as a function.

    Returns host_indices, frb_ra, frb_dec, frb_z, p_shell.
    """
    host_indices, p_shell = sample_stellar_mass_weighted_hosts(
        z_halo=z_halo,
        Mstar=Mstar,
        n_frb=n_frb,
        z_source=z_source,
        dz=dz,
        alpha_star=alpha_star,
        seed=seed,
    )

    ra_halo = np.asarray(ra_halo, dtype=float)
    dec_halo = np.asarray(dec_halo, dtype=float)
    if ra_halo.shape != np.asarray(z_halo).shape or dec_halo.shape != np.asarray(z_halo).shape:
        raise ValueError("ra_halo, dec_halo, and z_halo must have the same shape.")

    frb_ra = ra_halo[host_indices]
    frb_dec = dec_halo[host_indices]
    frb_z = np.full(host_indices.size, float(z_source))

    shell_mask = fixed_redshift_shell_mask(z_halo, z_source, dz=dz)
    plot_stellar_mass_host_histogram(
        Mstar_shell=np.asarray(Mstar)[shell_mask],
        Mstar_selected=np.asarray(Mstar)[host_indices],
        save_path=histogram_path,
    )

    return host_indices, frb_ra, frb_dec, frb_z, p_shell


# Pasteable notebook example:
#
# from frb_map_generation.stellar_mass_weighted_hosts import (
#     sample_stellar_mass_weighted_hosts,
#     fixed_redshift_shell_mask,
#     plot_stellar_mass_host_histogram,
# )
#
# z_source = 1.0
# host_indices, p_shell = sample_stellar_mass_weighted_hosts(
#     z_halo=z_halo,
#     Mstar=Mstar,
#     n_frb=10000,
#     z_source=z_source,
#     dz=0.02,
#     alpha_star=1.0,
#     seed=42,
# )
#
# frb_ra = ra_halo[host_indices]
# frb_dec = dec_halo[host_indices]
# frb_z = np.full(host_indices.size, z_source)
#
# shell_mask = fixed_redshift_shell_mask(z_halo, z_source, dz=0.02)
# plot_stellar_mass_host_histogram(
#     Mstar_shell=Mstar[shell_mask],
#     Mstar_selected=Mstar[host_indices],
#     save_path="stellar_mass_weighted_host_histogram.png",
# )
