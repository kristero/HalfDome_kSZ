"""Loader of the overlap/covariance particle-pass products (plan Step 1.2) and per-halo normalisations.

Product set (one directory; the real files appear in the study root, the synthetic rehearsal fixture with
the identical schema is ``tests/rehearsal_work/study/``):

  measurements_owner.npz      owner200 [n, 17 owners, 3 phases, 16 quantities, 67 X200 shells],
                              owner500 [.., 40 X500 shells], aux200/aux500 [n, 2 AGN, 3 aux, S], hs200/hs500
                              [n, S, 384] smoothing-length histograms, x200, x500, hs_edges
  measurements_excision.npz   excision [n, 4 variants, 4 zones, 3 members, 2 AGN, 5 reduced, 67]
  measurements_partition.npz  partition [n, 2 partition owners, 2 painted-bound, 2 AGN, 5 reduced, 67]
  measurements_legacy.npz     sums [n, 4 radii234 components, 11 radii234 sums, 36], edges, sum_names
  apertures.npz               apertures [n, 4, 2, 7], overlapper [n_ov, 2, 2, 6] (rows follow ov_ptr), ...
  selected_halos.npz          per-halo catalogue (rows 0..n_old-1 = radii234 in its order), weights, classes
  overlap_tables.npz          CSR overlapper / owner / excision / partition tables, excision_volume_fraction
  census_pairs.npz, census_population.npz, metadata_plan.json, schema.json, study_config.json

Axis names come from ``schema.json`` (``NewData.names``).  The study-root modules ``grids``, ``geometry``
and ``measure_overlap`` are loaded read-only from their exact paths (``study_modules``).

Normalisations (per halo, physical units):
  P200     = G M200c rho200 f_b / (2 R200c)                 (radii234 ``pilot_physics.p200``)
  rho_c    = M200c / (4 pi/3 R200c^3) / 200                (radii234 definition), f_b from metadata_plan.json
  electrons: n_e V mu_XGP / (V rho_c f_b)  (radii234 electron_components; MU_XGP_MSUN)
  B24 (R500c-scaled, Braspenning et al. 2024): f_B = 0.159, mu = 0.59, mu_e = 1.14,
           k T500 = mu m_p G M500 / (2 R500),  n_e,500 = 500 f_B rho_c / (mu_e m_p),  P_e,500 = n_e,500 k T500
  A10 (option): P500 = 1.65e-3 E(z)^(8/3) [M500 / (3e14 h70^-1 Msun)]^(2/3) h70^2 keV cm^-3
  self_similar_500 (option): G M500 rho500 f_b / (2 R500) (the radii234 form at Delta = 500)
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import _legacy

STUDY_ROOT = _legacy.STUDY_ROOT
FIXTURE_DIR = STUDY_ROOT / "tests" / "rehearsal_work" / "study"
PRODUCT_FILES = ("measurements_owner.npz", "measurements_excision.npz", "measurements_partition.npz",
                 "measurements_legacy.npz", "apertures.npz", "selected_halos.npz", "overlap_tables.npz",
                 "census_pairs.npz", "census_population.npz", "metadata_plan.json", "schema.json",
                 "study_config.json")
ARRAYS = {
    "owner200": ("measurements_owner.npz", "owner200"), "owner500": ("measurements_owner.npz", "owner500"),
    "aux200": ("measurements_owner.npz", "aux200"), "aux500": ("measurements_owner.npz", "aux500"),
    "hs200": ("measurements_owner.npz", "hs200"), "hs500": ("measurements_owner.npz", "hs500"),
    "x200": ("measurements_owner.npz", "x200"), "x500": ("measurements_owner.npz", "x500"),
    "hs_edges": ("measurements_owner.npz", "hs_edges"),
    "excision": ("measurements_excision.npz", "excision"),
    "kappa_zone_edges": ("measurements_excision.npz", "kappa_zone_edges"),
    "partition": ("measurements_partition.npz", "partition"),
    "legacy_sums": ("measurements_legacy.npz", "sums"), "legacy_edges": ("measurements_legacy.npz", "edges"),
    "legacy_sum_names": ("measurements_legacy.npz", "sum_names"),
    "apertures": ("apertures.npz", "apertures"), "inertia": ("apertures.npz", "inertia"),
    "id_hash": ("apertures.npz", "id_hash"), "id_count": ("apertures.npz", "id_count"),
    "diagnostics": ("apertures.npz", "diagnostics"), "overlapper": ("apertures.npz", "overlapper"),
    "overlapper_ptr": ("apertures.npz", "overlapper_ptr"),
}
# B24 (Braspenning et al. 2024) and A10 (Arnaud et al. 2010) conventions
B24 = {"f_B": 0.159, "mu": 0.59, "mu_e": 1.14}
A10_P500_KEV_CM3 = 1.65e-3
KEV_CM3_IN_PA = 1.602176634e-16 / 1e-6
# legacy (radii234) component -> owner codes (STEP1_README): central 0; own satellites odd 1-11 + 13;
# other haloes even 2-12 + 14 + 15; unbound 16
LEGACY_COMPONENT_OWNERS = {0: (0,), 1: (1, 3, 5, 7, 9, 11, 13), 2: (2, 4, 6, 8, 10, 12, 14, 15), 3: (16,)}
CLASS_NAMES = ("clean", "cont_reach1R", "cont_annulus_only")


def study_modules() -> SimpleNamespace:
    """grids, geometry, measure_overlap (and their pilot_physics/measure) from the study root, read-only."""
    root = STUDY_ROOT
    P = _legacy.load("overlap_pilot_physics", root / "pilot_physics.py")
    M = _legacy.load("overlap_measure", root / "measure.py", provides={"pilot_physics": P})
    G = _legacy.load("overlap_grids", root / "grids.py", provides={"pilot_physics": P})
    GEO = _legacy.load("overlap_geometry", root / "geometry.py")
    MO = _legacy.load("overlap_measure_overlap", root / "measure_overlap.py",
                      provides={"pilot_physics": P, "measure": M, "grids": G})
    return SimpleNamespace(P=P, M=M, G=G, GEO=GEO, MO=MO)


class NewData:
    """Lazy access to one product set; arrays are read on first use and cached.

    ``directory`` holds the product files (default: the study root); ``files`` maps a product file name to
    another path (e.g. a staged copy).  ``rows`` of every per-halo array follow ``selected_halos.npz``.
    """

    def __init__(self, directory: Path | str | None = None, files: dict | None = None):
        self.directory = Path(directory) if directory is not None else STUDY_ROOT
        self.paths = {name: self.directory / name for name in PRODUCT_FILES}
        self.paths.update({k: Path(v) for k, v in (files or {}).items()})
        self.mod = study_modules()
        self.schema = json.loads(self.paths["schema.json"].read_text())
        self.names = {k: tuple(v) if isinstance(v, list) else v for k, v in self.schema["names"].items()}
        self.plan = json.loads(self.paths["metadata_plan.json"].read_text())
        self.config = json.loads(self.paths["study_config.json"].read_text())
        self.halos = dict(np.load(self.paths["selected_halos.npz"]))
        self.tables = dict(np.load(self.paths["overlap_tables.npz"]))
        self.n = len(self.halos["m200c_msun"])
        self.n_old = int(self.config["old_sample"]["rows"][1])
        self.baryon_fraction = float(self.plan["baryon_fraction"])
        self.softening_mpc = float(self.plan["softening_physical_mpc"])
        self._cache: dict = {}
        self._npz: dict = {}
        self._check_names()

    # ------------------------------------------------------------------ arrays
    def array(self, key: str) -> np.ndarray:
        if key not in self._cache:
            fname, member = ARRAYS[key]
            if fname not in self._npz:
                self._npz[fname] = np.load(self.paths[fname])
            self._cache[key] = self._npz[fname][member]
        return self._cache[key]

    def release(self, *keys) -> None:
        """Drop cached arrays (memory) -- all if no key is given."""
        for k in (keys or list(self._cache)):
            self._cache.pop(k, None)

    def census(self, which: str = "pairs") -> dict:
        key = f"census_{which}"
        if key not in self._cache:
            self._cache[key] = dict(np.load(self.paths[f"census_{which}.npz"]))
        return self._cache[key]

    def _check_names(self) -> None:
        MO = self.mod.MO
        expected = {"owners": MO.OWNERS, "phases": MO.PHASES, "quantities": MO.QUANTITIES, "reduced": MO.REDUCED,
                    "variants": MO.VARIANTS, "zones": MO.ZONES, "members": MO.MEMBERS, "agn": MO.AGN,
                    "partition_owners": MO.PARTITION_OWNERS, "painted_bound": MO.PAINTED_BOUND}
        bad = [k for k, v in expected.items() if tuple(self.names.get(k, ())) != tuple(v)]
        if bad:
            raise ValueError(f"schema.json names differ from measure_overlap for {bad}")

    def index(self, axis: str, label) -> int:
        return list(self.names[axis]).index(label)

    # ------------------------------------------------------------------ grids
    def grid(self, base: str) -> np.ndarray:
        """Stored X200 / X500 edges (the shell assignment used them); checked against grids.py."""
        stored = np.asarray(self.array("x200" if base == "x200" else "x500"), float)
        local = self.mod.G.X200 if base == "x200" else self.mod.G.X500
        if stored.shape != local.shape or not np.allclose(stored, local, rtol=1e-12, atol=0):
            raise ValueError(f"stored {base} grid differs from grids.py")
        return stored

    def subgrid_index(self, grid) -> tuple[str, np.ndarray]:
        """(base, indices into the stored base edges) for a grid name or (base, edges) pair.

        Names: 'x200' (67), 'old' (36, radii234), 'lee22' (27 over 0.04-1.34), 'x500' (40), 'b24' (30 over
        0.01-3 R500c); custom: ('x200' | 'x500', edges) with edges an exact subset of grids.py's grid.
        Indices are computed on grids.py's grid and applied to the stored grid (robust to ULP differences
        between the machine that wrote the products and this one).
        """
        G = self.mod.G
        if isinstance(grid, str):
            table = {"x200": ("x200", np.arange(G.N200 + 1)), "old": ("x200", G.OLD_INDEX),
                     "lee22": ("x200", G.LEE22_INDEX), "x500": ("x500", np.arange(G.N500 + 1)),
                     "b24": ("x500", G.B24_INDEX)}
            if grid not in table:
                raise ValueError(f"unknown grid {grid!r}")
            return table[grid]
        base, edges = grid
        local = G.X200 if base == "x200" else G.X500
        return base, G.subgrid_index(local, np.asarray(edges, float))

    def edges(self, grid) -> np.ndarray:
        base, idx = self.subgrid_index(grid)
        return self.grid(base)[idx]

    @staticmethod
    def rebin(values: np.ndarray, idx: np.ndarray, axis: int = -1) -> np.ndarray:
        """Exact linear rebinning onto the sub-grid given by edge indices (np.add.reduceat)."""
        v = np.moveaxis(np.asarray(values), axis, -1)[..., idx[0]:idx[-1]]
        out = np.add.reduceat(v, idx[:-1] - idx[0], axis=-1)
        return np.moveaxis(out, -1, axis)

    # ------------------------------------------------------------------ per-halo scales
    def _rows(self, rows):
        return slice(None) if rows is None else np.asarray(rows)

    def radius(self, base: str = "x200", rows=None) -> np.ndarray:
        key = "r_200_crit_mpc" if base == "x200" else "r_500_crit_mpc"
        return np.asarray(self.halos[key], float)[self._rows(rows)]

    def mass(self, base: str = "x200", rows=None) -> np.ndarray:
        key = "m200c_msun" if base == "x200" else "m500c_msun"
        return np.asarray(self.halos[key], float)[self._rows(rows)]

    def rho_c(self, base: str = "x200", rows=None) -> np.ndarray:
        """Critical density from the halo's own SO mass and radius (radii234 definition), Msun/Mpc^3."""
        delta = 200.0 if base == "x200" else 500.0
        m, r = self.mass(base, rows), self.radius(base, rows)
        return m / (4 * np.pi / 3 * r**3) / delta

    def p200(self, rows=None) -> np.ndarray:
        """radii234 P200 [Pa] (pilot_physics.p200 with this study's f_b)."""
        return self.mod.P.p200(self.mass("x200", rows), self.radius("x200", rows), self.baryon_fraction)

    def p500(self, convention: str = "B24", rows=None) -> np.ndarray:
        """P_e,500 [Pa]: 'B24' (n_e,500 k T500), 'A10' (Arnaud et al. 2010), 'self_similar_500'."""
        P = self.mod.P
        m, r = self.mass("x500", rows), self.radius("x500", rows)
        if convention == "B24":
            return self.ne500("B24", rows) * self.kT500(rows)
        if convention == "A10":
            h70 = self.hubble() / 0.7
            ez = self.e_of_z()
            return (A10_P500_KEV_CM3 * ez**(8 / 3) * (m * h70 / 3e14)**(2 / 3) * h70**2) * KEV_CM3_IN_PA
        if convention == "self_similar_500":
            return P.p200(m, r, self.baryon_fraction)
        raise ValueError(convention)

    def kT500(self, rows=None) -> np.ndarray:
        """k T500 = mu m_p G M500 / (2 R500) [J] (B24 mu = 0.59)."""
        P = self.mod.P
        m = self.mass("x500", rows) * P.MSUN_KG
        r = self.radius("x500", rows) * P.MPC_M
        return B24["mu"] * P.proton_mass * P.G * m / (2 * r)

    def ne500(self, convention: str = "B24", rows=None) -> np.ndarray:
        """n_e,500 = 500 f_B rho_c / (mu_e m_p) [m^-3] (B24 f_B = 0.159, mu_e = 1.14)."""
        P = self.mod.P
        if convention != "B24":
            raise ValueError("n_e,500 is defined in the B24 convention only")
        rho = self.rho_c("x500", rows) * P.MSUN_KG / P.MPC_M**3
        return 500 * B24["f_B"] * rho / (B24["mu_e"] * P.proton_mass)

    def hubble(self) -> float:
        return float(np.atleast_1d(self.plan["cosmology"]["h"])[0])

    def e_of_z(self) -> float:
        c = self.plan["cosmology"]
        z = float(self.plan.get("redshift", 0.0))
        om = float(np.atleast_1d(c.get("Omega_m", [0.3]))[0])
        ol = float(np.atleast_1d(c.get("Omega_lambda", [1 - om]))[0])
        return float(np.sqrt(om * (1 + z)**3 + ol + (1 - om - ol) * (1 + z)**2))

    def inclusion_weights(self, rows=None) -> np.ndarray:
        """1 / inclusion probability of the (bin, class) stratum (selected_halos 'weight')."""
        return np.asarray(self.halos["weight"], float)[self._rows(rows)]

    def mass_bin(self, rows=None) -> np.ndarray:
        return np.asarray(self.halos["mass_bin"])[self._rows(rows)]

    # ------------------------------------------------------------------ census classes
    def overlapper_table(self) -> dict:
        """Per-overlapper columns (ov_*) with their halo row ('halo')."""
        t = self.tables
        ptr = t["ov_ptr"]
        out = {k[3:]: t[k] for k in t if k.startswith("ov_") and k != "ov_ptr"}
        out["halo"] = np.repeat(np.arange(self.n), np.diff(ptr))
        out["ptr"] = ptr
        return out

    def overlapper_select(self, mu_min: float = 0.1, reach: str = "any", kind: str = "all") -> np.ndarray:
        """Boolean over overlapper rows: mu >= mu_min, reach ('any' = inside 2R, 'reach1R', 'annulus'),
        kind ('all', 'own_satellites', 'neighbours', 'neighbour_centrals')."""
        ov = self.overlapper_table()
        sel = ov["mu"] >= mu_min
        if reach == "reach1R":
            sel &= ov["reach1"]
        elif reach == "annulus":
            sel &= ~ov["reach1"]
        elif reach != "any":
            raise ValueError(reach)
        own = ov["own_satellite"].astype(bool)
        if kind == "own_satellites":
            sel &= own
        elif kind == "neighbours":
            sel &= ~own
        elif kind == "neighbour_centrals":
            sel &= ~own & ov["is_central"].astype(bool)
        elif kind != "all":
            raise ValueError(kind)
        return sel

    def census_class(self, mu_min: float = 0.1, kind: str = "all") -> np.ndarray:
        """0 clean, 1 cont_reach1R (an overlapper reaches inside 1R), 2 cont_annulus_only (1-2R only).

        Derived from the overlapper lists (all of which reach inside 2R, mu >= 0.03); at mu_min = 0.1 and
        kind = 'all' it must equal selected_halos 'census_class_main'.
        """
        if mu_min < 0.03 - 1e-12:
            raise ValueError("overlapper lists hold mu >= 0.03 only")
        ov = self.overlapper_table()
        sel = self.overlapper_select(mu_min, "any", kind)
        any2 = np.bincount(ov["halo"][sel], minlength=self.n) > 0
        any1 = np.bincount(ov["halo"][sel & ov["reach1"]], minlength=self.n) > 0
        return np.where(any1, 1, np.where(any2, 2, 0)).astype(np.int8)

    def census_class_from_max_mu(self, mu_min: float = 0.1, kind: str = "all") -> np.ndarray:
        """Same classes from selected_halos 'census_max_mu' [n, (reach1R, reach2R), (own, neighbour)]."""
        mx = np.asarray(self.halos["census_max_mu"], float)
        cols = {"all": [0, 1], "own_satellites": [0], "neighbours": [1]}[kind]
        thr = np.float32(mu_min)       # census_max_mu is float32
        r1 = (mx[:, 0, cols] >= thr).any(axis=1)
        r2 = (mx[:, 1, cols] >= thr).any(axis=1)
        return np.where(r1, 1, np.where(r2, 2, 0)).astype(np.int8)

    # ------------------------------------------------------------------ legacy view
    def legacy_dataset(self) -> dict:
        """measurements_legacy.npz as an analysis.io data dict (radii234 normalisation, legacy edges)."""
        from . import io as aio
        P = self.mod.P
        return {"sums": self.array("legacy_sums"), "axes": ("halo", "component", "quantity", "shell"),
                "coords": {"component": aio.COMPONENTS, "quantity": tuple(P.SUM_NAMES)},
                "edges": np.asarray(self.array("legacy_edges"), float), "model_edges": np.asarray(P.EDGES, float),
                "radius_mpc": self.radius("x200"), "mass_msun": self.mass("x200"), "overdensity": 200.0,
                "baryon_fraction": self.baryon_fraction, "softening_mpc": self.softening_mpc,
                "halos": self.halos, "stored": {}, "grid": "legacy 36"}


def load(directory=None, files=None) -> NewData:
    return NewData(directory, files)


def load_fixture() -> NewData:
    return NewData(FIXTURE_DIR)
