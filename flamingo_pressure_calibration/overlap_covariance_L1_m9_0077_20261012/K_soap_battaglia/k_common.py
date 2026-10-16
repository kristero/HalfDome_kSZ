"""Shared configuration of Test K (Battaglia B12/B16 refit to FLAMINGO L1_m9 HYDRO_FIDUCIAL SOAP-HBT, all outputs).

Environment (set by run_test_k.py): K_MODE 'full' | 'smoke'; K_RESULTS (results dir); K_SCRATCH (node-local per-halo
intermediates); K_WORKERS; STUDY_ROOT (study directory).

Physical conventions (k_fit_core docstring has the profile forms):
  x = r / R200c, physical radii; R200c, M200c from SOAP SO/200_crit (SORadius is comoving: converted with the
  'Conversion factor to physical CGS (including cosmological corrections)' attribute, i.e. times a);
  P200 = G M200c (200 rho_c(z)) f_b / (2 R200c) with 200 rho_c(z) = M200c / (4 pi R200c^3 / 3) (per halo, exact by
  definition of the SO radius); f_b = Omega_b / Omega_m(incl. massive neutrinos) = 0.0486 / 0.306 = 0.15882 (the study
  value, Tests B-J);
  Ytilde(<r) = Y(<r) m_e c^2 / sigma_T / (P200 R200c^3) (Y = SOAP ComptonY[WithoutRecentAGNHeating], physical Mpc^2);
  Mtilde(<r) = M_gas(<r) / (f_b M200c).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

MODE = os.environ.get("K_MODE", "full")
SMOKE = MODE == "smoke"
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("STUDY_ROOT", str(HERE.parent))).resolve()
RESULTS = Path(os.environ.get("K_RESULTS", str(HERE / ("results_smoke" if SMOKE else "results")))).resolve()
SCRATCH = Path(os.environ.get("K_SCRATCH", str(RESULTS / "work"))).resolve()
WORKERS = int(os.environ.get("K_WORKERS", "1" if SMOKE else "8"))

SOAP_DIR = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT"
SNAPS = tuple(range(7, 78))                         # halo_properties_0007 ... 0077 (71 files)
SMOKE_SNAPS = (77, 37)
LOG_MMIN = 12.7                                     # central haloes with M200c >= 10^12.7 Msun
MASS_EDGES = np.round(np.arange(12.7, 15.71, 0.2), 2)   # 0.2-dex cells (15 bins to 15.7)
N_BOOT = 200                                        # bootstrap resamples of haloes per cell
N_SUB = 256                                         # random per-cell subsample stored for checks and figures
N_STRATA_M, N_STRATA_X = 4, 8                       # quadrature nodes per (cell, aperture): mass x radius strata
                                                    # (smoke test on 256-halo subsamples: max |ln error| 0.08-0.20%
                                                    # for 4 x 8 vs 0.21-0.51% for 4 x 4)
SEED = 20261015
F_B = 0.0486 / 0.306                                # = 0.158824; checked against the SOAP header in K1
MPC_CM = 3.08567758e24
MPC_M = MPC_CM / 100.0
MSUN_G = 1.98841e33                                 # SOAP / SWIFT solar mass
MSUN_KG = MSUN_G / 1e3
G_SI = 6.67430e-11
SIGMA_T = 6.6524587321e-29
ME_C2 = 8.1871057769e-14                            # m_e c^2 [J]
SOFTENING_MPC = 0.0057                              # maximal physical softening of L1_m9 (study value)

SO = ("2500_crit", "1000_crit", "500_crit", "200_crit", "100_crit", "BN98", "200_mean", "50_crit", "5xR_500_crit")
KPC = (10, 30, 50, 100, 300, 500, 1000, 3000)
AP_NAMES = tuple(f"SO/{a}" for a in SO) + tuple(f"IS/{k}kpc" for k in KPC)
N_SO, N_AP = len(SO), len(SO) + len(KPC)
# columns read per output (full column, then the selected rows)
COLS_BASE = ["InputHalos/IsCentral", "SO/200_crit/TotalMass", "SO/500_crit/TotalMass", "SO/200_crit/Concentration",
             "InputHalos/NumberOfBoundParticles"]
COLS_SO = [f"SO/{a}/{q}" for a in SO for q in ("SORadius", "ComptonYWithoutRecentAGNHeating", "ComptonY", "GasMass",
                                                "NumberOfGasParticles")]
COLS_KPC = [f"InclusiveSphere/{k}kpc/{q}" for k in KPC for q in ("GasMass", "NumberOfGasParticles")]
COLUMNS = COLS_BASE + COLS_SO + COLS_KPC

# usability of an aperture in a cell (all must hold)
MIN_MEDIAN_GAS = 50          # median NumberOfGasParticles over the cell's haloes
MIN_VALID_FRAC = 0.9         # fraction of the cell's haloes for which the aperture is filled
MIN_X = 0.025                # mean aperture radius >= 0.025 R200c
MIN_X_SPHERE = 0.1           # physical spheres enter the gas fit only at mean x >= 0.1 and radius >= 100 kpc: inside,
MIN_R_KPC_GAS = 100          # the SOAP gas mass is dominated by the central galaxy's cold / star-forming gas (smoke
                             # run: 30-50 kpc hold 2-25x the published B16 hot-gas profile; first full fit: mean
                             # ln(SOAP/law) +0.51 at 30 kpc and +0.07 at 50 kpc where x >= 0.1 at high z; SOAP audit:
                             # ionisation deficit 6-25% below 0.1 R200c); field 'Gcore' keeps the spheres >= 30 kpc
                             # down to x = 0.025 as a check
MIN_R_KPC = 4 * SOFTENING_MPC * 1e3     # physical sphere radius >= 4 softening lengths (22.8 kpc): drops 10 kpc
MIN_CELL_HALOES = 10         # cells with fewer central haloes are not used
DEDUP_LN = 0.05              # within a cell, an aperture whose mean x is within 5% of a kept one is dropped
SO_PRIORITY = ("SO/200_crit", "SO/500_crit", "SO/2500_crit", "SO/1000_crit", "SO/5xR_500_crit", "SO/50_crit",
               "SO/100_crit", "SO/200_mean", "SO/BN98")
# fit ranges: SO apertures by name (whatever their mean x in the cell: 50c lies at x ~ 1.8, 5xR500c at x ~ 3.3), kpc
# spheres by the cell's mean x <= xmax (1 / 1.8 / 3.3 as decided for Test K; run 2 used 1.9 / 3.6, undocumented)
RANGES = {
    "R1": {"label": "x <= 1 (within R200c; spheres with mean x <= 1)", "so": ("SO/2500_crit", "SO/1000_crit", "SO/500_crit", "SO/200_crit"),
           "xmax": 1.0},
    "R2": {"label": "x <~ 1.8 (adds 100c, BN98, 200m, 50c; spheres with mean x <= 1.8)",
           "so": ("SO/2500_crit", "SO/1000_crit", "SO/500_crit", "SO/200_crit", "SO/100_crit", "SO/BN98", "SO/200_mean",
                  "SO/50_crit"), "xmax": 1.8},
    "R3": {"label": "x <~ 3.3 (adds 5xR500c; spheres with mean x <= 3.3; neighbour gas included)",
           "so": tuple(f"SO/{a}" for a in SO), "xmax": 3.3},
}
PARAMETRIZATIONS = ("orig", "brk_amp", "brk_all", "free")
FIELDS = {"P": {"quantity": "Yn", "form": "B12", "label": "pressure (Y without recent AGN heating)"},
          "G": {"quantity": "Mg", "form": "B16", "label": "gas density (gas mass)"},
          "Pagn": {"quantity": "Ya", "form": "B12", "label": "pressure check (Y with recent AGN heating)"},
          "Gcore": {"quantity": "Mg", "form": "B16", "label": "gas density check with the spheres down to 0.025 R200c (central-galaxy gas)"}}
DATA_FIELD = {"P": "P", "Pagn": "P", "G": "G", "Gcore": "G"}     # data units of each fit field
N_BOOT_FIT = 100 if not SMOKE else 4              # bootstrap refits of the 'orig' P and G fits
N_BOOT_OTHER = 40 if not SMOKE else 4             # ... of the brk_amp / brk_all / free fits
N_BOOT_CHECK = 20 if not SMOKE else 2             # ... of the Pagn / Gcore check fits
if os.environ.get("K_BOOT"):                       # override "orig,other,check" (run_test_k.py --boot)
    N_BOOT_FIT, N_BOOT_OTHER, N_BOOT_CHECK = (int(v) for v in os.environ["K_BOOT"].split(","))
RESUME = os.environ.get("K_RESUME") == "1"         # K3: re-use nominal fits / bootstrap chunks already in work/


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=_default) + "\n")
    tmp.replace(path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(path):
    return json.loads(Path(path).read_text())


def cell_label(snap, b):
    return f"{snap:04d}_{MASS_EDGES[b]:.1f}-{MASS_EDGES[b + 1]:.1f}"
