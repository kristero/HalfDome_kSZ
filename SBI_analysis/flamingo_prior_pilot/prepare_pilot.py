#!/usr/bin/env python3
"""Audit the full proposed box and compress the actual catalogue for proposals."""
import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc
import toml

from prior_model import (FIDUCIAL, HIGH, LOW, NAMES, central_column,
    domain_metrics, finite_pressure_integral, from_unit, parameters, save_json)


def catalogue_histogram(catalogue, campaign, output):
    rng = toml.load(campaign / "code/local_operator_probe.toml")["rng_uniform"]
    cz = 2*rng[0]-1
    phi = 2*np.pi*rng[1]
    centre = np.array([np.sqrt(1-cz*cz)*np.cos(phi), np.sqrt(1-cz*cz)*np.sin(phi), cz])
    radius, apo = np.arccos(.2), np.pi/180
    mass_edges = np.linspace(12, 16, 41)
    z_edges = np.linspace(0, np.log1p(4), 49)
    count = np.zeros((40, 48))
    mass_sum = count.copy()
    z_sum = count.copy()
    unmasked_count = count.copy()
    individual = []
    support = dict(count=0, mass_min=1e100, mass_max=0., z_min=1e100, z_max=0.)
    with h5py.File(catalogue, "r") as h:
        total = len(h["redshift"])
        for first in range(0, total, 1000000):
            indices = slice(first, min(first+1000000, total))
            mass = np.asarray(h["halo_mass_m200c"][indices], dtype=float)/.68
            z = np.asarray(h["redshift"][indices], dtype=float)
            ds = h["Position"]
            position = np.asarray(ds[indices, :3] if ds.shape[0] == total else ds[:3, indices].T, dtype=float)
            keep = np.isfinite(mass) & np.isfinite(z) & (mass >= 1e12) & (z >= 0)
            mass, z, position = mass[keep], z[keep], position[keep]
            support["count"] += len(mass)
            for key, value in (("mass_min", mass.min()), ("z_min", z.min())):
                support[key] = min(support[key], float(value))
            for key, value in (("mass_max", mass.max()), ("z_max", z.max())):
                support[key] = max(support[key], float(value))
            angle = np.arccos(np.clip(position.dot(centre)/np.linalg.norm(position, axis=1), -1, 1))
            weight = np.zeros(len(mass))
            weight[angle <= radius-apo] = 1
            taper = (angle > radius-apo) & (angle < radius)
            weight[taper] = .5*(1+np.cos(np.pi*(angle[taper]-radius+apo)/apo))
            weight *= weight
            # Resolve the nearest massive haloes individually; their angular
            # extent and mass matter more than their histogram bin centres.
            bright = (mass >= 3e14) & (z < .15) & (weight > 0)
            if bright.any():
                individual.append(np.column_stack([mass[bright], z[bright], weight[bright]]))
            logm, logz = np.log10(mass), np.log1p(z)
            unmasked_count += np.histogram2d(logm, logz, (mass_edges, z_edges))[0]
            weight[bright] = 0
            for target, w in ((count, weight), (mass_sum, weight*logm), (z_sum, weight*logz)):
                target += np.histogram2d(logm, logz, (mass_edges, z_edges), weights=w)[0]
            print("Catalogue audit: {}/{}".format(first+len(mass), total), flush=True)
    if support["mass_max"] > 1e16 or support["z_max"] > 4:
        raise ValueError("Catalogue lies outside histogram domain")
    occupied = count > 0
    points = np.column_stack([10**(mass_sum[occupied]/count[occupied]),
                              np.expm1(z_sum[occupied]/count[occupied]), count[occupied]])
    if individual:
        points = np.vstack([points]+individual)
    np.savez(output / "catalogue_quadrature.npz", mass=points[:, 0], z=points[:, 1],
             weight=points[:, 2], mass_edges=mass_edges, log1pz_edges=z_edges,
             unmasked_count=unmasked_count, mask_center=centre)
    support.update(quadrature_points=len(points),
        individually_resolved_halos=sum(len(p) for p in individual),
        mask_approximation="Continuous cap weight at halo centres, squared; exact pixel mask used in full maps")
    save_json(output / "catalogue_support.json", support)


def make_design(output):
    random = from_unit(qmc.Sobol(9, scramble=True, seed=20260914).random_base2(16))
    corners = from_unit(np.array(list(itertools.product((0., 1.), repeat=9))))
    examples = [("Battaglia12", "control", FIDUCIAL.copy())]
    for index, name in enumerate(NAMES):
        for side, value in (("low", LOW[index]), ("high", HIGH[index])):
            theta = FIDUCIAL.copy()
            theta[index] = value
            examples.append((name+"_"+side, "single_edge", theta))
    for first, second in itertools.combinations(range(9), 2):
        for a, b in itertools.product((0, 1), repeat=2):
            theta = FIDUCIAL.copy()
            theta[first] = (LOW, HIGH)[a][first]
            theta[second] = (LOW, HIGH)[b][second]
            examples.append((NAMES[first]+str(a)+"__"+NAMES[second]+str(b), "pair_edge", theta))
    params = np.array([p[2] for p in examples])
    np.savez(output / "prior_design.npz", theta=random, corners=corners,
             examples=params, example_names=np.array([p[0] for p in examples]),
             example_types=np.array([p[1] for p in examples]), parameter_names=NAMES)
    all_results = {}
    for label, theta in (("sobol", random), ("corners", corners), ("examples", params)):
        metrics = domain_metrics(theta)
        # Evaluate finite-radius integrated pressure over a modest physical grid.
        mg, zg = np.meshgrid(10**np.array([12., 13., 14., 15., 15.7]), [0., .5, 1., 2., 4., 5.])
        mass, z = mg.ravel(), zg.ravel()
        pf, xf, bf = parameters(FIDUCIAL, mass, z)
        y_fid = finite_pressure_integral(pf, xf, bf)
        ylow, yhigh, tail = [], [], []
        for start in range(0, len(theta), 256):
            p0, xc, beta = parameters(theta[start:start+256], mass, z)
            ratio = finite_pressure_integral(p0, xc, beta)/y_fid
            ylow.extend(ratio.min(axis=1))
            yhigh.extend(ratio.max(axis=1))
            # At r=0, estimate missed column beyond the current 1e5 R200
            # integration boundary whenever an untruncated integral exists.
            tail_here = np.ones_like(beta)
            good = beta > .7
            from scipy.special import betainc
            tail_here[good] = 1-betainc(.7, beta[good]-.7, 1e5/(1e5+xc[good]))
            tail.extend(tail_here.max(axis=1))
        metrics.update(min_Y200_ratio=np.array(ylow), max_Y200_ratio=np.array(yhigh),
                       max_missing_central_column=np.array(tail))
        np.savez(output / (label+"_audit.npz"), **metrics)
        all_results[label] = dict(count=len(theta),
            los_divergent=int(np.sum(metrics["min_beta"] <= .7)),
            infinite_thermal_energy_divergent=int(np.sum(metrics["min_beta"] <= 2.7)),
            central_column_tail_gt_1pct=int(np.sum(metrics["max_missing_central_column"] > .01)),
            Y200_above_100x_fiducial_somewhere=int(np.sum(metrics["max_Y200_ratio"] > 100)),
            Y200_below_0p01x_fiducial_somewhere=int(np.sum(metrics["min_Y200_ratio"] < .01)),
            min_Y200_ratio=float(min(ylow)), max_Y200_ratio=float(max(yhigh)))
    save_json(output / "prior_audit.json", dict(groups=all_results,
        interpolation_domain=dict(logmass=[12, 15.7], redshift=[.001, 5]),
        interpretation="Y200 ratio thresholds flag large deviations; they are not observational exclusion limits"))
    # Check analytic central columns against independent direct integration.
    from scipy.integrate import quad
    errors = []
    for xc in (.1, .497, 4):
        for beta in (.8, 1.5, 2.8, 4.35, 16):
            analytic = central_column(1., xc, beta)
            numerical = 2*quad(lambda t: np.exp(-.3*np.log(t/xc)-beta*np.log1p(t/xc)),
                               0, np.inf, epsabs=0, epsrel=1e-8, limit=400)[0]
            errors.append(abs(numerical/analytic-1))
    if max(errors) > 1e-6:
        raise ValueError("Central pressure integral regression failed")
    save_json(output / "analytic_validation.json", dict(cases=len(errors),
              max_relative_error=max(errors), status="passed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "audit"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "catalogue_support.json").exists():
        catalogue_histogram(args.catalogue, args.campaign, output)
    if not (output / "analytic_validation.json").exists():
        make_design(output)
    print("Prior and catalogue preparation completed", flush=True)


if __name__ == "__main__":
    main()
