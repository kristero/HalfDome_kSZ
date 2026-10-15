"""Per-halo profile variants from the new particle-pass products (Tests A, B, D, F).

    prof = variants.profile(data, field, estimator='geometric', agn='removed', gas_selection='all',
                            grid='x200', excision=None, template=None, normalisation='auto', rows=None)

returns a ``Profile`` with ``values`` [halo, shell] in normalised units, ``edges`` (in units of the grid's
radius), ``retained_fraction`` (excision) and the linear ingredient sums.  Linear sums are always rebinned
BEFORE any ratio is formed (never average ratios).

field / estimator (pressure; Pe = y / V m_e c^2 / sigma_T per particle):
  'geometric'  sum y / V_shell                        (tSZ/DM-correct default; radii234 op order)
  'Q_V'        sum V Pe / sum V = electron_kT / volume (n_e k T, SPH-volume weighted; B24 Q_V)
  'Q_V_y'      sum y m_e c^2/sigma_T / sum V           (same with the y-based Pe)
  'Q_M'        mass_pe / mass                           (mass weighted)
  'Q_X'        lx_pe / lx_rosat over the X-ray-selected phase
  'nT_X'       geometric n_e x k T_X, T_X = lx_t / lx_rosat (X-ray phase)
electrons: 'geometric' (sum n_e V / V_shell), 'Q_V' (sum n_e V / sum V); gas: 'geometric', 'Q_V'
(sum m / sum V), 'Q_M' (sum m rho / sum m; clumping-biased).  ``diagnostic(...)``: 'T_M', 'T_V', 'T_X' [K],
'clumping_density' (<rho^2>_V/<rho>_V^2), 'clumping_pressure' (<Pe^2>_V/<Pe>_V^2).

agn: 'removed' = phases xray_selected + other (the default, = radii234 y_noagn), 'kept' = all phases.
Excision and partition blocks carry an AGN axis instead of phases (not_recent_agn / recent_agn).

gas_selection (``selection(...)``): 'all'; radii234 controls 'no_other', 'no_satellites_or_other',
'host_bound'; 'central_bound'; overlapper removal ``selection('remove_overlappers', mu_min=0.1,
reach='any'|'reach1R'|'annulus', flavour='remove_all'|'keep_own_satellites')`` (keep-own-satellites
removes neighbours only); A2-i 'remove_bound_to_other_painted' (owner 14 = gas bound to another painted
distinct halo outside the owner table, plus, per halo, every neighbour-overlapper owner code whose
overlappers are painted -- by bound-gas-weighted majority, see ``a2i_ambiguity``); A2-ii
'partition_owned' (scaled-distance partition owner = central; reduced quantities only) and
'partition_owned_not_other_painted' (A2-ii and the partition block's bound-to-other-painted flag).

excision = {'variant': 'all'|'neighbours_only'|'placebo_iso'|'placebo_nbr', 'kappa': 0.5|1|2,
            'members': 'all'|'no_overlapper_bound'|'central_bound', 'min_retained': 0.5}:
retained zones s >= kappa; value = retained sum / (V_shell (1 - f_exc)) with f_exc from
overlap_tables excision_volume_fraction; shells retaining < min_retained of their volume are NaN.

template = {'model': F(M, x) -> P/P200 (or n_e mu/(f_b rho_c), rho/(f_b rho_c)) of an overlapper of mass M
            at x = r/R_j, 'mu_min': 0.1, 'reach': 'any', 'kind': 'all'|'neighbour_centrals'|'neighbours'|
            'own_satellites', 'truncation': 1.0 (in R_j), 'amplitude': 'model'|'measured_satellites'|
            'measured_all', 'n_seg': 600}:
subtracts sum_k <F_k>_shell, the exact angular + shell average of each overlapper's model profile centred
at its offset (geometry.offset_shell_average), with the overlapper's own R_j (R200c for centrals,
pseudo-R200c for satellites) and its own P200 from (M_j, R_j); 'measured_*' rescales the amplitude so the
model's integral inside its truncation sphere equals the overlapper's measured bound y / electrons / mass
(apertures 'overlapper' block, whole sphere; own bound for satellites, group bound for centrals).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .newdata import NewData

FIELDS = ("pressure", "electrons", "gas")
ESTIMATORS = {"pressure": ("geometric", "Q_V", "Q_V_y", "Q_M", "Q_X", "nT_X"),
              "electrons": ("geometric", "Q_V"), "gas": ("geometric", "Q_V", "Q_M")}
DIAGNOSTICS = ("T_M", "T_V", "T_X", "clumping_density", "clumping_pressure")
LEGACY_SELECTIONS = {"all": (0, 1, 2, 3), "no_other": (0, 1, 3), "no_satellites_or_other": (0, 3),
                     "host_bound": (0, 1), "central_bound": (0,)}
ZONE_FOR_KAPPA = {0.5: 1, 1.0: 2, 2.0: 3}       # retained zones are >= this index (zone edges 0.5/1/2)


# ============================================================================ selections
@dataclass(frozen=True)
class Selection:
    """Gas selection: an owner mask (owner blocks), a per-halo owner mask, or partition cells."""
    name: str
    owners: tuple | None = None                 # 17 booleans (same for every halo)
    per_halo: str | None = None                 # 'a2i_painted': per-halo owner masks from the tables
    partition_owner: tuple = (0, 1)
    painted_bound: tuple = (0, 1)
    members: tuple | None = None                # excision members allowed with this selection
    label: str = ""

    @property
    def uses_partition(self) -> bool:
        return self.owners is None and self.per_halo is None


def _owner_tuple(codes) -> tuple:
    mask = np.zeros(17, bool)
    mask[list(codes)] = True
    return tuple(bool(x) for x in mask)


def _owner_code(mu_class: int, annulus: int, neighbour: int) -> int:
    return 1 + (mu_class * 2 + annulus) * 2 + neighbour


def selection(name: str = "all", mu_min: float = 0.1, reach: str = "any", flavour: str = "remove_all",
              include_neighbour_overlappers: bool = True) -> Selection:
    """Build a gas selection (see module docstring)."""
    from .newdata import LEGACY_COMPONENT_OWNERS as LC
    if name in LEGACY_SELECTIONS:
        codes = [c for comp in LEGACY_SELECTIONS[name] for c in LC[comp]]
        members = {"all": (0, 1, 2), "central_bound": (0,)}.get(name)
        return Selection(name, owners=_owner_tuple(codes), members=members, label=name)
    if name == "remove_overlappers":
        classes = [k for k, t in enumerate((0.03, 0.1, 0.3)) if t >= mu_min - 1e-12]
        if not classes or mu_min not in (0.03, 0.1, 0.3):
            raise ValueError("mu_min must be one of the census classes 0.03, 0.1, 0.3")
        reaches = {"any": (0, 1), "reach1R": (0,), "annulus": (1,)}[reach]
        kinds = {"remove_all": (0, 1), "keep_own_satellites": (1,)}[flavour]
        removed = {_owner_code(m, r, t) for m in classes for r in reaches for t in kinds}
        codes = [c for c in range(17) if c not in removed]
        return Selection(f"remove_overlappers(mu>={mu_min},{reach},{flavour})", owners=_owner_tuple(codes),
                         label=f"remove overlapper-bound gas: mu >= {mu_min}, reach {reach}, {flavour}")
    if name == "remove_bound_to_other_painted":
        if include_neighbour_overlappers:
            return Selection(name, per_halo="a2i_painted",
                             label="A2-i: gas bound to other painted distinct haloes removed (owner 14 and the "
                                   "neighbour-overlapper owner codes whose overlappers are painted)")
        return Selection(name + "(owner14_only)", owners=_owner_tuple([c for c in range(17) if c != 14]),
                         label="A2-i (owner 14 only): gas bound to other painted haloes outside the owner table")
    if name == "partition_owned":
        return Selection(name, owners=None, partition_owner=(0,), painted_bound=(0, 1),
                         label="A2-ii: scaled-distance partition owner = central")
    if name == "partition_owned_not_other_painted":
        return Selection(name, owners=None, partition_owner=(0,), painted_bound=(0,),
                         label="A2-ii and A2-i: partition owner = central and not bound to another painted halo "
                               "(partition flag; outside the owner table)")
    raise ValueError(f"unknown selection {name!r}")


def _as_selection(sel) -> Selection:
    if isinstance(sel, Selection):
        return sel
    if isinstance(sel, dict):
        return selection(**sel)
    return selection(sel)


# ============================================================================ profile container
@dataclass
class Profile:
    values: np.ndarray                 # [halo, shell] normalised
    edges: np.ndarray                  # in units of the grid radius
    grid: object
    base: str                          # 'x200' or 'x500'
    field: str
    estimator: str
    agn: str
    selection: str
    normalisation: str
    rows: np.ndarray
    retained_fraction: np.ndarray | None = None
    sums: dict = dc_field(default_factory=dict)
    template: np.ndarray | None = None
    info: dict = dc_field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.values)


# ============================================================================ sums from blocks
def _phases(data: NewData, agn: str, only_xray: bool = False) -> np.ndarray:
    names = data.names["phases"]
    if only_xray:
        return np.array([p == "xray_selected" for p in names], float)
    if agn == "removed":
        return np.array([p != "recent_agn" for p in names], float)
    if agn == "kept":
        return np.ones(len(names))
    raise ValueError(agn)


def _agn_axis(data: NewData, agn: str) -> list:
    return [0] if agn == "removed" else [0, 1]


def owner_sum(data: NewData, base: str, quantity: str, sel: Selection, agn: str, rows=None,
              only_xray: bool = False) -> np.ndarray:
    """Linear sum [halo, base shell] of one owner-block quantity over the selected owners and phases."""
    block = data.array("owner200" if base == "x200" else "owner500")
    qi = data.index("quantities", quantity)
    sub = block[:, :, :, qi, :] if rows is None else block[np.asarray(rows)][:, :, :, qi, :]
    phases = _phases(data, agn, only_xray)
    if sel.per_halo is not None:
        masks = owner_masks(data, sel, rows).astype(float)
        return np.einsum("hops,ho,p->hs", sub, masks, phases)
    return np.einsum("hops,o,p->hs", sub, np.asarray(sel.owners, float), phases)


def painted_overlappers(data: NewData) -> np.ndarray:
    """Per overlapper row: True if the overlapper (or, for a satellite, its host) is a painted distinct halo
    (host M200c >= painting threshold; census_population painted_by_hid)."""
    flags = np.asarray(data.census("population")["painted_by_hid"])
    hid = np.asarray(data.tables["ov_hid"], np.int64)
    out = np.zeros(len(hid), bool)
    known = (hid >= 0) & (hid < len(flags))
    out[known] = flags[hid[known]] == 1
    return out


def owner_masks(data: NewData, sel: Selection, rows=None) -> np.ndarray:
    """[halo, 17] owner masks (per-halo selections; constant selections broadcast)."""
    rows = np.arange(data.n) if rows is None else np.asarray(rows)
    if sel.per_halo is None:
        return np.tile(np.asarray(sel.owners, bool), (len(rows), 1))
    if sel.per_halo != "a2i_painted":
        raise ValueError(sel.per_halo)
    ov = data.overlapper_table()
    painted = painted_overlappers(data)
    code = 1 + (np.asarray(ov["mu_class"]) * 2 + (~ov["reach1"].astype(bool)).astype(int)) * 2 \
        + (~ov["own_satellite"].astype(bool)).astype(int)
    gas = np.asarray(ov["bound_gas_mass_msun"], float)
    masks = np.ones((data.n, 17), bool)
    masks[:, 14] = False
    neighbour = ~ov["own_satellite"].astype(bool)
    # bound-gas-weighted share of painted overlappers per (halo, neighbour owner code)
    key = ov["halo"] * 17 + code
    w_all = np.bincount(key[neighbour], weights=gas[neighbour] + 1e-30, minlength=data.n * 17).reshape(data.n, 17)
    w_painted = np.bincount(key[neighbour & painted], weights=gas[neighbour & painted] + 1e-30,
                            minlength=data.n * 17).reshape(data.n, 17)
    remove = (w_all > 0) & (w_painted > 0.5 * w_all)
    masks[remove] = False
    return masks[rows]


def a2i_ambiguity(data: NewData) -> dict:
    """How often a neighbour owner code mixes painted and unpainted overlappers (A2-i decides by bound gas)."""
    ov = data.overlapper_table()
    painted = painted_overlappers(data)
    neighbour = ~ov["own_satellite"].astype(bool)
    code = 1 + (np.asarray(ov["mu_class"]) * 2 + (~ov["reach1"].astype(bool)).astype(int)) * 2 + neighbour.astype(int)
    key = (ov["halo"] * 17 + code)[neighbour]
    n_all = np.bincount(key, minlength=data.n * 17)
    n_painted = np.bincount(key[painted[neighbour]], minlength=data.n * 17)
    mixed = (n_painted > 0) & (n_painted < n_all)
    return {"neighbour_overlappers": int(neighbour.sum()), "painted_neighbour_overlappers": int((neighbour & painted).sum()),
            "halo_code_cells": int((n_all > 0).sum()), "mixed_cells": int(mixed.sum()),
            "haloes_with_mixed_cells": int(np.unique(np.flatnonzero(mixed.reshape(data.n, 17).any(axis=1))).size)}


def partition_sum(data: NewData, quantity: str, sel: Selection, agn: str, rows=None) -> np.ndarray:
    block = data.array("partition")
    qi = data.index("reduced", quantity)
    sub = block if rows is None else block[np.asarray(rows)]
    sub = sub[:, list(sel.partition_owner)][:, :, list(sel.painted_bound)][:, :, :, _agn_axis(data, agn)][..., qi, :]
    return sub.sum(axis=(1, 2, 3))


def excision_sum(data: NewData, quantity: str, variant: str, kappa: float, members, agn: str, rows=None) -> np.ndarray:
    block = data.array("excision")
    qi = data.index("reduced", quantity)
    v = data.index("variants", variant)
    first_zone = ZONE_FOR_KAPPA[float(kappa)]
    sub = block[:, v] if rows is None else block[np.asarray(rows)][:, v]
    sub = sub[:, first_zone:][:, :, list(members)][:, :, :, _agn_axis(data, agn)][..., qi, :]
    return sub.sum(axis=(1, 2, 3))


def excision_fraction(data: NewData, variant: str, kappa: float, rows=None) -> np.ndarray:
    """Excised geometric volume fraction [halo, X200 shell] (float64)."""
    t = data.tables
    v = data.index("variants", variant)
    k = int(np.flatnonzero(np.isclose(np.asarray(t["kappa"], float), float(kappa)))[0])
    f = np.asarray(t["excision_volume_fraction"][:, v, k, :], float)
    return f if rows is None else f[np.asarray(rows)]


# ============================================================================ normalisation
def _volume(data: NewData, base: str, edges: np.ndarray, rows) -> np.ndarray:
    """Geometric shell volumes [halo, shell] in physical Mpc^3 (radii234 op order)."""
    radius = data.radius(base, rows)
    return 4 * np.pi / 3 * radius[:, None]**3 * np.diff(np.asarray(edges, float)**3)


def _scales(data: NewData, base: str, normalisation: str, rows) -> dict:
    """Per-halo normalising scales: pressure [Pa], number density [Mpc^-3] and mass density [Msun Mpc^-3]."""
    P = data.mod.P
    fb = data.baryon_fraction
    if normalisation == "auto":
        normalisation = "P200" if base == "x200" else "B24"
    if normalisation == "P200":
        rho_c = data.rho_c("x200", rows)
        return {"name": "P200", "pressure": data.p200(rows), "rho_c": rho_c, "fb": fb,
                "electrons_per_mpc3": rho_c * fb / P.MU_XGP_MSUN, "mass_per_mpc3": rho_c * fb}
    if normalisation in ("B24", "A10", "self_similar_500"):
        rho_c = data.rho_c("x500", rows)
        ne500_mpc3 = data.ne500("B24", rows) * P.MPC_M**3
        from .newdata import B24
        return {"name": normalisation, "pressure": data.p500(normalisation, rows), "rho_c": rho_c,
                "electrons_per_mpc3": ne500_mpc3, "mass_per_mpc3": 500 * B24["f_B"] * rho_c}
    raise ValueError(normalisation)


def _normalise_pressure(pa: np.ndarray, scales: dict) -> np.ndarray:
    return pa / scales["pressure"][:, None]


def _pressure_geometric(data, y, volume, scales):
    """y [Mpc^2] over V_shell [Mpc^3] -> Pa -> normalised, in radii234's operation order."""
    P = data.mod.P
    with np.errstate(divide="ignore", invalid="ignore"):      # fully excised shells -> NaN (masked later)
        out = y * P.electron_mass * P.c**2 / P.SIGMA_T / (volume * P.MPC_M)
        if scales["name"] == "P200":
            out /= scales["pressure"][:, None]
            return out
        return out / scales["pressure"][:, None]


def _density_geometric(data, linear, volume, scales, kind):
    """Electrons (n_e V) or mass over V_shell, normalised (radii234 op order for P200 scaling)."""
    P = data.mod.P
    with np.errstate(divide="ignore", invalid="ignore"):
        if scales["name"] == "P200":
            rho_c = scales["rho_c"][:, None]
            if kind == "electrons":
                return linear * P.MU_XGP_MSUN / (volume * rho_c * scales["fb"])
            return linear / (volume * rho_c * scales["fb"])
        norm = scales["electrons_per_mpc3" if kind == "electrons" else "mass_per_mpc3"][:, None]
        return linear / volume / norm


# ============================================================================ the constructor
def profile(data: NewData, field: str, estimator: str = "geometric", agn: str = "removed",
            gas_selection="all", grid="x200", excision: dict | None = None, template: dict | None = None,
            normalisation: str = "auto", rows=None) -> Profile:
    if field not in FIELDS:
        raise ValueError(field)
    if estimator not in ESTIMATORS[field]:
        raise ValueError(f"estimator {estimator!r} not available for {field}")
    sel = _as_selection(gas_selection)
    base, idx = data.subgrid_index(grid)
    edges = data.grid(base)[idx]
    rows_arr = np.arange(data.n) if rows is None else np.asarray(rows)
    scales = _scales(data, base, normalisation, rows)
    volume = _volume(data, base, edges, rows)
    rb = lambda a: data.rebin(a, idx)      # noqa: E731
    reduced_only = excision is not None or sel.uses_partition
    if reduced_only and (estimator not in ("geometric", "Q_V") or (field == "pressure" and estimator != "geometric")):
        raise ValueError("excision and partition selections carry the reduced quantities only "
                         "(count, mass, volume, y, electrons): geometric estimators (and Q_V for densities)")
    if (excision is not None or sel.uses_partition) and base != "x200":
        raise ValueError("excision and partition blocks exist on the X200 grid only")
    retained = None
    sums = {}

    def linear(quantity, only_xray=False):
        if excision is not None:
            members = _excision_members(sel, excision)
            return rb(excision_sum(data, quantity, excision.get("variant", "all"), excision.get("kappa", 1.0),
                                   members, agn, rows))
        if sel.uses_partition:
            return rb(partition_sum(data, quantity, sel, agn, rows))
        return rb(owner_sum(data, base, quantity, sel, agn, rows, only_xray))

    if excision is not None:
        f = excision_fraction(data, excision.get("variant", "all"), excision.get("kappa", 1.0), rows)
        v200 = _volume(data, "x200", data.grid("x200"), rows)
        kept = rb(v200 * (1 - f))
        retained = kept / volume
        volume_eff = kept
    else:
        volume_eff = volume
    P = data.mod.P
    if field == "pressure":
        if estimator == "geometric":
            y = linear("y")
            sums["y"] = y
            values = _pressure_geometric(data, y, volume_eff, scales)
        elif estimator == "Q_V":
            ekt, vol = linear("electron_kT"), linear("volume")
            sums.update(electron_kT=ekt, volume=vol)
            values = _ratio(ekt, vol * P.MPC_M**3) / scales["pressure"][:, None]
        elif estimator == "Q_V_y":
            y, vol = linear("y"), linear("volume")
            sums.update(y=y, volume=vol)
            values = _ratio(y * data.mod.MO.PE_FACTOR, vol) / scales["pressure"][:, None]
        elif estimator == "Q_M":
            mpe, m = linear("mass_pe"), linear("mass")
            sums.update(mass_pe=mpe, mass=m)
            values = _ratio(mpe, m) / scales["pressure"][:, None]
        elif estimator == "Q_X":
            lpe, lx = linear("lx_pe", True), linear("lx_rosat", True)
            sums.update(lx_pe=lpe, lx_rosat=lx)
            values = _ratio(lpe, lx) / scales["pressure"][:, None]
        else:   # nT_X
            ne, lt, lx = linear("electrons"), linear("lx_t", True), linear("lx_rosat", True)
            sums.update(electrons=ne, lx_t=lt, lx_rosat=lx)
            n_e = ne / (volume * P.MPC_M**3)
            values = n_e * P.Boltzmann * _ratio(lt, lx) / scales["pressure"][:, None]
    else:
        q = "electrons" if field == "electrons" else "mass"
        if estimator == "geometric":
            s = linear(q)
            sums[q] = s
            values = _density_geometric(data, s, volume_eff, scales, field)
        elif estimator == "Q_V":
            s, vol = linear(q), linear("volume")
            sums.update({q: s, "volume": vol})
            values = _density_geometric(data, s, np.where(vol > 0, vol, np.nan), scales, field)
        else:   # Q_M (gas): sum m rho / sum m
            mr, m = linear("mass_rho"), linear("mass")
            sums.update(mass_rho=mr, mass=m)
            values = _ratio(mr, m) / scales["mass_per_mpc3"][:, None]
    if excision is not None:
        values = np.where(retained >= excision.get("min_retained", 0.5), values, np.nan)
    tmpl = None
    if template is not None:
        if estimator != "geometric":
            raise ValueError("templates are subtracted from geometric profiles only")
        tmpl = template_profiles(data, field, base, edges, scales, template, agn, rows_arr)
        values = values - tmpl
    return Profile(values=values, edges=edges, grid=grid, base=base, field=field, estimator=estimator, agn=agn,
                   selection=sel.name, normalisation=scales["name"], rows=rows_arr, retained_fraction=retained,
                   sums=sums, template=tmpl,
                   info={"selection_label": sel.label, "excision": excision,
                         "template": None if template is None else {k: v for k, v in template.items() if k != "model"}})


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def _excision_members(sel: Selection, excision: dict):
    members = excision.get("members", "all")
    table = {"all": (0, 1, 2), "no_overlapper_bound": (0, 2), "central_bound": (0,)}
    if sel.name not in ("all", "central_bound"):
        raise ValueError("excision applies to the 'all' (or 'central_bound') selection; choose members instead")
    if sel.name == "central_bound":
        members = "central_bound"
    return table[members]


# ============================================================================ diagnostics
def diagnostic(data: NewData, kind: str, agn: str = "removed", gas_selection="all", grid="x200", rows=None) -> np.ndarray:
    """Temperatures [K] ('T_M', 'T_V', 'T_X') and clumping factors on any owner-block grid."""
    sel = _as_selection(gas_selection)
    base, idx = data.subgrid_index(grid)
    rb = lambda a: data.rebin(a, idx)      # noqa: E731
    q = lambda name, xray=False: rb(owner_sum(data, base, name, sel, agn, rows, xray))   # noqa: E731
    if kind == "T_M":
        return _ratio(q("mass_t"), q("mass"))
    if kind == "T_V":
        return _ratio(q("volume_t"), q("volume"))
    if kind == "T_X":
        return _ratio(q("lx_t", True), q("lx_rosat", True))
    if kind == "clumping_density":
        m, vol = q("mass"), q("volume")
        return _ratio(q("mass_rho") * vol, m * m)
    if kind == "clumping_pressure":
        vol = q("volume")
        pv = q("y") * data.mod.MO.PE_FACTOR           # sum V Pe
        return _ratio(q("volume_pe2") * vol, pv * pv)
    raise ValueError(kind)


# ============================================================================ resolution masks
def resolution(data: NewData, rows, grid="x200", **options) -> dict:
    """analysis.masks.resolution_masks on any grid of the new products: particle counts from the owner
    blocks (all owners and phases), upper-median smoothing lengths from the rebinned hs200/hs500
    histograms (h a / R200c or R500c), the grid's own radius for the softening criterion."""
    from . import io as aio
    from . import masks as msk
    base, idx = data.subgrid_index(grid)
    rows = np.asarray(rows)
    block = data.array("owner200" if base == "x200" else "owner500")
    qc = data.index("quantities", "count")
    count = data.rebin(block[rows][:, :, :, qc].sum(axis=(1, 2)), idx)
    hist = np.moveaxis(data.array("hs200" if base == "x200" else "hs500")[rows], -2, -1)    # [halo, nbin, shell]
    hist = np.moveaxis(data.rebin(hist.astype(np.int64), idx), -1, -2)
    hsml = aio.upper_median_from_hist(hist, data.array("hs_edges"))
    ds = {"sums": count[:, None, :], "axes": ("halo", "quantity", "shell"), "coords": {"quantity": ("count",)},
          "edges": data.grid(base)[idx], "radius_mpc": np.zeros(data.n), "softening_mpc": data.softening_mpc,
          "hsml_median_x": np.full((data.n, len(idx) - 1), np.nan)}
    ds["radius_mpc"][rows] = data.radius(base, rows)
    full_count = np.zeros((data.n, 1, len(idx) - 1))
    full_count[rows, 0] = count
    ds["sums"] = full_count
    ds["hsml_median_x"][rows] = hsml
    return msk.resolution_masks(ds, rows, **options)


# ============================================================================ excision helpers
def common_halo_set(*profiles: Profile) -> np.ndarray:
    """[halo, shell] True where every given profile is finite (fixed halo set per shell across methods)."""
    out = np.ones_like(profiles[0].values, bool)
    for p in profiles:
        out &= np.isfinite(p.values)
    return out


def excision_common_set(data: NewData, variants=("all",), kappas=(0.5, 1.0, 2.0), grid="x200",
                        min_retained: float = 0.5, rows=None) -> np.ndarray:
    """[halo, shell] True where every (variant, kappa) retains >= min_retained of the shell volume."""
    base, idx = data.subgrid_index(grid)
    if base != "x200":
        raise ValueError("excision exists on X200 only")
    edges = data.grid("x200")
    v200 = _volume(data, "x200", edges, rows)
    vol = data.rebin(v200, idx)
    out = np.ones(vol.shape, bool)
    for v in variants:
        for k in kappas:
            kept = data.rebin(v200 * (1 - excision_fraction(data, v, k, rows)), idx)
            out &= kept / vol >= min_retained
    return out


# ============================================================================ templates
def _norm_of(data: NewData, field: str, mass, radius):
    """Profile scale of an overlapper of mass M and radius R (its own 200c-type scaling)."""
    P = data.mod.P
    fb = data.baryon_fraction
    rho_c = np.asarray(mass, float) / (4 * np.pi / 3 * np.asarray(radius, float)**3) / 200.0
    if field == "pressure":
        return P.p200(mass, radius, fb)
    if field == "electrons":
        return rho_c * fb / P.MU_XGP_MSUN
    return rho_c * fb


def _volume_integral(f, cut: float, n_seg: int = 600, start: float = 1e-7) -> float:
    """int_0^cut 4 pi s^2 f(s) ds (log-spaced Gauss-Legendre; f integrable as s^p, p > -3)."""
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(6)
    s_edges = np.geomspace(start * cut, cut, n_seg + 1)
    a, b = s_edges[:-1, None], s_edges[1:, None]
    x = a + (b - a) * (nodes + 1) / 2
    return float(np.sum(4 * np.pi * x * x * f(x) * weights * (b - a) / 2))


def template_profiles(data: NewData, field: str, base: str, edges: np.ndarray, scales: dict, spec: dict, agn: str,
                      rows) -> np.ndarray:
    """[halo, shell] sum over the selected overlappers of their offset model profiles, in profile units."""
    GEO = data.mod.GEO
    model = spec["model"]
    trunc = float(spec.get("truncation", 1.0))
    amplitude = spec.get("amplitude", "model")
    n_seg = int(spec.get("n_seg", 600))
    sel = data.overlapper_select(spec.get("mu_min", 0.1), spec.get("reach", "any"), spec.get("kind", "all"))
    ov = data.overlapper_table()
    ptr = ov["ptr"]
    rows = np.asarray(rows)
    r_norm = data.radius(base, rows)
    norm_h = {"pressure": scales["pressure"], "electrons": scales["electrons_per_mpc3"],
              "mass": scales["mass_per_mpc3"]}["mass" if field == "gas" else field]
    P = data.mod.P
    out = np.zeros((len(rows), len(edges) - 1))
    measured = None
    if amplitude != "model":
        block = data.array("overlapper")
        qname = {"pressure": "y" if agn == "kept" else "y_noagn", "electrons": "electrons", "gas": "mass"}[field]
        measured = block[:, :, 1, data.index("overlapper_quantities", qname)]      # [ov, (own, group)]
    for i, h in enumerate(rows):
        for k in range(ptr[h], ptr[h + 1]):
            if not sel[k]:
                continue
            m_j, r_j, d_j = float(ov["mass_msun"][k]), float(ov["radius_mpc"][k]), float(ov["distance_mpc"][k])
            amp = float(_norm_of(data, field, m_j, r_j)) / float(norm_h[i])
            scale = r_norm[i] / r_j
            cut = trunc * r_j / r_norm[i]
            if measured is not None:
                own = bool(ov["own_satellite"][k]) or not bool(ov["is_central"][k])
                if amplitude == "measured_all" or (amplitude == "measured_satellites" and not ov["is_central"][k]):
                    target = float(measured[k, 0 if own else 1])
                    amp *= template_amplitude_scale(data, field, model, m_j, amp, scale, cut, r_norm[i],
                                                    float(norm_h[i]), target, n_seg)
            G = GEO.make_G(lambda s, a=amp, sc=scale, mj=m_j: a * model(mj, s * sc), cut, n_seg=n_seg)
            out[i] += GEO.offset_shell_average(G, edges, d_j / r_norm[i], cut)
    return out


def model_integral(data: NewData, field: str, model, m_j: float, amp: float, scale: float, cut: float,
                   r_norm: float, norm_h: float, n_seg: int = 600) -> float:
    """Volume integral of one template inside its truncation sphere, in the measured units:
    y [Mpc^2] (pressure), electron count (electrons) or Msun (gas)."""
    P = data.mod.P
    integral = _volume_integral(lambda s: amp * model(m_j, s * scale), cut, n_seg) * r_norm**3 * norm_h
    if field == "pressure":
        integral *= P.SIGMA_T / (P.electron_mass * P.c**2) * P.MPC_M
    return integral


def template_amplitude_scale(data: NewData, field: str, model, m_j, amp, scale, cut, r_norm, norm_h, target,
                             n_seg: int = 600) -> float:
    """Factor making the template's integral equal ``target`` (the overlapper's measured bound gas)."""
    integral = model_integral(data, field, model, m_j, amp, scale, cut, r_norm, norm_h, n_seg)
    return target / integral if integral > 0 else 1.0


# ============================================================================ stacks
def stack(values: np.ndarray, weights=None, mask=None) -> dict:
    """Per-shell (weighted) mean over haloes with finite values (and ``mask``); returns mean, n, n_eff."""
    v = np.asarray(values, float)
    ok = np.isfinite(v) if mask is None else (np.isfinite(v) & np.asarray(mask, bool))
    w = np.ones(v.shape[0]) if weights is None else np.asarray(weights, float)
    W = np.where(ok, w[:, None], 0.0)
    total = W.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(total > 0, (W * np.where(ok, v, 0.0)).sum(axis=0) / total, np.nan)
        neff = np.where(total > 0, total**2 / (W**2).sum(axis=0), 0.0)
    return {"mean": mean, "n": ok.sum(axis=0), "n_eff": neff}


# ============================================================================ checks against radii234
def legacy_component_selection_names():
    return tuple(LEGACY_SELECTIONS)


def compare_with_legacy(data: NewData, legacy: dict | None = None, rows=None) -> dict:
    """Max relative deviation of the new-block profiles (old 36-shell grid) from a radii234-layout dataset.

    ``legacy``: an analysis.io data dict, sums [halo, 4, 11, 36], whose halo rows are ``rows`` of this
    product set (default: this set's measurements_legacy.npz, all rows).  Covers pressure (all gas and
    no-AGN), electrons and gas for every radii234 membership selection.  Normalisation is identical on both
    sides (this product set's R200c, M200c and f_b).
    """
    from . import io as aio
    legacy = data.legacy_dataset() if legacy is None else legacy
    rows = np.arange(len(legacy["sums"])) if rows is None else np.asarray(rows)
    out = {}
    pairs = [("pressure", "kept", "pressure"), ("pressure", "removed", "pressure_noagn"),
             ("electrons", "kept", "electrons"), ("gas", "kept", "gas")]
    for sel in ("all", "no_other", "no_satellites_or_other", "host_bound", "central_bound"):
        legacy_sel = sel if sel != "central_bound" else (0,)
        for field, agn, io_field in pairs:
            new = profile(data, field, "geometric", agn, sel, "old", rows=rows).values
            old = aio.profile(legacy, io_field, legacy_sel)
            ref = np.abs(old).max(axis=-1, keepdims=True)
            out[f"{sel}|{io_field}"] = float(np.max(np.abs(new - old) / np.maximum(ref, 1e-300)))
    return out


def radii234_reference_dataset(data: NewData, measurements_path, rows=None) -> dict:
    """A radii234 measurements.npz (or the rehearsal radii234_reference_measurements.npz) as an analysis.io
    dataset for rows 0..n_old-1 of this product set, normalised with THIS set's halo scales and f_b.

    On the real data use radii234_contamination_L1_m9_0077_20261008/measurements.npz: its stored normalised
    arrays used radii234's f_b; ``info['baryon_fraction_ratio']`` reports new/old if the plan is available.
    """
    with np.load(measurements_path) as z:
        sums = z["sums"]
        edges = z["edges"] if "edges" in z.files else data.array("legacy_edges")
    n = len(sums)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    ds = data.legacy_dataset()
    ds.update(sums=sums, edges=np.asarray(edges, float), radius_mpc=data.radius("x200", rows),
              mass_msun=data.mass("x200", rows), grid="radii234 reference")
    return ds
