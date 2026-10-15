#!/usr/bin/env python3
"""Check periodic shell pruning against an exhaustive small-box construction."""

import itertools
import json

import numpy as np
from scipy.integrate import quad

from build_lightcone import distance_interpolators, intersecting_tiles
from preflight import H_TNG, OUTPUT


def main():
    generator = np.random.default_rng(8127)
    points = np.vstack((np.zeros((1, 3)), np.full((1, 3), 1 - 1e-12), generator.uniform(size=(30, 3))))
    observer = np.array([0.02, 0.51, 0.98])
    edges = np.array([0, 0.35, 1.2, 2.7])
    all_shells = set()
    for lower, upper in zip(edges[:-1], edges[1:]):
        expected, actual = set(), set()
        for tiles, result in ((itertools.product(range(-4, 5), repeat=3), expected),
                              (intersecting_tiles(1., observer, lower, upper), actual)):
            for tile in tiles:
                radius = np.linalg.norm(points + np.asarray(tile) - observer, axis=1)
                for index in np.flatnonzero((radius >= lower) & (radius < upper)):
                    result.add((tuple(tile), int(index)))
        assert expected == actual, "Bounding-box pruning missed or added halo centres"
        assert all_shells.isdisjoint(actual), "Adjacent shells double-count a halo image"
        all_shells.update(actual)
    chi, inverse = distance_interpolators()
    errors = []
    for redshift in (0.001, 0.05, 0.5, 0.99729422578194, 1.0):
        exact = quad(lambda z: 299792.458 / (100 * H_TNG * np.sqrt(0.3089 * (1 + z)**3 + 0.6911)),
                     0, redshift, epsabs=1e-10)[0]
        errors.append(abs(float(chi(redshift)) / exact - 1))
        assert abs(float(inverse(chi(redshift))) - redshift) < 1e-10
    assert max(errors) < 1e-8
    result = dict(periodic_shell_vs_exhaustive="pass", adjacent_shell_duplicates=0,
                  checked_halo_images=len(all_shells), maximum_distance_relative_error=max(errors))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "geometry_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
