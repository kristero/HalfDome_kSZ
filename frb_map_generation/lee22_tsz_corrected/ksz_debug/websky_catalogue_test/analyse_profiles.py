# Profile/construction part of the HalfDome-vs-WebSky halo kSZ gap: WebSky catalogue (and HalfDome) one-halo power with
#  - our XGPaint B16 (cache), B16 with the eq.-A1 exponent as printed, B16 with Stein's alpha=1/gamma=-0.3,
#  - the pks2map profile as transcribed by Y. Guan (external, unverified), with its SIS M200c proxy,
#  - WebSky's halo-map selection (M200c > 1e13, theta200c > 0.5') and Delta=3 uniform-sphere compensation (Stein+20 3.2.1).
# Also: where the first z > 4.5 halo sits in the mass-sorted catalogue (pks2map velocity-index issue, external report).
import os, json
import numpy as np, h5py

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_catalogue_test"
PROD = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002"
T2 = 2.7255e6**2; OM, OB, H = 0.31, 0.049, 0.68; FB = OB / OM
SIGMA_T = 6.6524587e-29; M_P = 1.67262192e-27; MSUN = 1.98847e30; MPC = 3.0856775814913673e22
RHOM_COM = 2.775e11 * OM * H**2; ARCMIN = np.pi / 180 / 60
LREP = np.array([300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0])

tab = h5py.File(os.path.join(OUT, "tau_ell_table_b16.h5"), "r")
ell = tab["ell"][()]; le = tab["logm_edges"][()]; ze = tab["z_edges"][()]
lmc = 0.5 * (le[1:] + le[:-1]); zc = 0.5 * (ze[1:] + ze[:-1]); Mc = 10**lmc
TH200 = tab["theta200c_rad"][()].T; dA = tab["dA_mpc"][()]; chi = dA * (1 + zc)
PROF = {"xgpaint_b16": (tab["a_b16_x4p0"][()].transpose(2, 1, 0), tab["a0_b16_x4p0"][()].T)}
u = h5py.File(os.path.join(OUT, "tau_ell_table_independent.h5"), "r")
PROF["indep_b16"] = (u["a_b16_indep_x4p0"][()].transpose(2, 1, 0), u["a0_b16_indep_x4p0"][()].T)
PROF["pks2map"] = (u["a_pks2map_x4p0"][()].transpose(2, 1, 0), u["a0_pks2map_x4p0"][()].T)
vfile = os.path.join(OUT, "tau_ell_table_b16_variants.h5")
if os.path.exists(vfile):
    v = h5py.File(vfile, "r")
    PROF["b16_printed_exponent"] = (v["a_b16_printed_x4p0"][()].transpose(2, 1, 0), v["a0_b16_printed_x4p0"][()].T)
    PROF["b16_alpha1_gamma03"] = (v["a_b16_a1g03_x4p0"][()].transpose(2, 1, 0), v["a0_b16_a1g03_x4p0"][()].T)
NL, NM, NZ = PROF["xgpaint_b16"][0].shape

def cl1h(v2, a): return np.einsum("mz,lmz->l", v2, a * a) / (4 * np.pi)
def dl(c): return ell * (ell + 1) * c / (2 * np.pi) * T2
def at(c, L=LREP): return np.exp(np.interp(np.log(L), np.log(ell), np.log(np.maximum(c, 1e-300))))
dig = np.loadtxt(os.path.join(PROD, "websky_halo_stein2020_fig6_digitized_binned.csv"), delimiter=",")
wsd = np.exp(np.interp(np.log(LREP), np.log(dig[:, 0]), np.log(dig[:, 1]), left=np.nan, right=np.nan))

with h5py.File(os.path.join(PROD, "maps", "halo_mass_redshift_histogram.h5"), "r") as f:
    hd_v2 = f["sum_vlos_over_c_squared"][()].T
ws = np.load(os.path.join(OUT, "websky_halo_mass_redshift_histograms.npz"))
def remap(h, shift):
    out = np.zeros_like(h); w = 0.025
    for iz in range(NZ):
        s = shift[iz] / w; k = int(np.floor(s)); fr = s - k
        for j in np.nonzero(h[:, iz])[0]:
            for jj, ww in ((j + k, 1 - fr), (j + k + 1, fr)):
                if 0 <= jj < NM and ww > 0: out[jj, iz] += ww * h[j, iz]
    return out
def omz(z): return OM * (1 + z)**3 / (OM * (1 + z)**3 + 1 - OM)
def delta_bn(z): x = omz(z) - 1; return 18 * np.pi**2 + 82 * x - 39 * x**2
ws_sis_v2 = remap(ws["m200m_v2"].astype(float), np.log10(np.sqrt(delta_bn(zc) / 200)))
CAT = {"websky_c7m": ws["c7m_v2"].astype(float), "websky_sis": ws_sis_v2, "halfdome": hd_v2}
sel = (Mc[:, None] > 1e13) & (TH200 > 0.5 * ARCMIN)

def w_sphere(x): x = np.maximum(x, 1e-6); return 3 * (np.sin(x) - x * np.cos(x)) / x**3
R3 = (3 * Mc[:, None] / (4 * np.pi * 3.0 * RHOM_COM))**(1 / 3) * np.ones((1, NZ))
TH3 = R3 / chi[None, :]
NE = 1.0 / (1.136 * M_P)
AMP_WE = SIGMA_T * FB * Mc[:, None] * MSUN * NE / (dA[None, :] * MPC)**2
def comp(a, a0, kind):
    if kind == "none": return a
    amp = AMP_WE if kind == "websky" else a0
    return a - amp[None] * w_sphere(ell[:, None, None] * TH3[None])

rows = {}
for pname, (a, a0) in PROF.items():
    for cname, v2 in CAT.items():
        for cut in (False, True):
            for kind in ("none", "websky", "zero"):
                key = f"{cname}|{pname}|{'cut' if cut else 'nocut'}|comp_{kind}"
                d = at(dl(cl1h(v2 * (sel if cut else 1.0), comp(a, a0, kind))))
                rows[key] = {"D": d.tolist(), "over_websky_dig": (d / wsd).tolist()}
# gas ledger of each profile within 4 R200c relative to f_b M200c (1.136 m_p per electron)
ledger = {p: {f"logM{lm}_z{z}": float(PROF[p][1][np.argmin(abs(lmc - lm)), np.argmin(abs(zc - z))] /
                                      AMP_WE[np.argmin(abs(lmc - lm)), np.argmin(abs(zc - z))])
              for lm in (13.0, 14.0, 15.0) for z in (0.1, 0.5, 1.0, 2.0)} for p in PROF}
# first z > 4.5 halo in the mass-sorted WebSky file (every-1000th subsample, file order)
sub = ws["subsample_every1000"]
first = int(np.argmax(sub[:, 1] > 4.5)); nz45 = np.cumsum(sub[:, 1] > 4.5)
frac_m13_shifted = float(np.mean(nz45[sub[:, 0] * 0.8 > 1e13] > 0))
out = {"ell_rep": LREP.tolist(), "websky_digitized": wsd.tolist(), "rows": rows, "gas_ledger_over_fbM200c": ledger,
       "first_z_gt_4.5_subsample_index_x1000": first * 1000, "M200m_at_first_z_gt_4.5": float(sub[first, 0]),
       "fraction_of_subsample_with_z_gt_4.5": float(np.mean(sub[:, 1] > 4.5))}
json.dump(out, open(os.path.join(OUT, "profile_variants_summary.json"), "w"), indent=1)
np.set_printoptions(precision=3, suppress=True, linewidth=220)
print("ell", LREP, "\nWebSky dig", wsd)
for k, r in rows.items():
    if ("comp_zero" in k) and ("halfdome" in k): continue
    print(f"{k:62s} D={np.array(r['D'])}  /dig={np.array(r['over_websky_dig'])}")
print(json.dumps(ledger, indent=0))
print("first z>4.5 halo at file index ~", first * 1000, "M200m", sub[first, 0], "; fraction z>4.5", out["fraction_of_subsample_with_z_gt_4.5"])
