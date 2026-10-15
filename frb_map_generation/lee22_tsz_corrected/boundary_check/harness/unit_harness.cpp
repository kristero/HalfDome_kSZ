/*
 * unit_harness.cpp -- periodic-boundary unit harness for group_particles
 * (L. Thiele, github.com/leanderthiele/group_particles), the C++ library with
 * which Lee et al. 2022 (arXiv:2205.01710, MNRAS 517, 420) extracted the TNG300
 * electron pressure / density profiles.
 *
 * The harness is compiled against the REAL library headers of one tree
 * (exported by build_and_run.sh; same g++ flags as the repository's compile.sh):
 *   detail/workspace_sorting.hpp  Sorting: GRID clamp, compute_offsets, prt_idx_ranges,
 *                                 PER(x) hint macro, sph_cub_intersect
 *   detail/geom_utils.hpp         periodic_idx, periodic_hypotsq (with and without hint),
 *                                 periodic_dist_whint
 *   include/fields.hpp            coord_t (= float)
 * No library source is modified.  The private nested class Workspace<AF>::Sorting
 * is reached through the access exemption for names in explicit instantiations.
 *
 * For synthetic TNG300-geometry groups and particles (Bsize = 205000 ckpc/h,
 * 64^3 cells, float coordinates, R = 2.5 R200c for 1e13-1e15 Msun/h at z = 0-2,
 * comoving ckpc/h) it evaluates every (group, particle) pair with
 *   hint  : sorted loop, Rsq = periodic_hypotsq(rgrp, rprt, Bsize, periodic_to_add)
 *           = default code path at c3fe9b8 (Lee22 era; no NAIVE, no EARLY_RETURN)
 *   plain : sorted loop, Rsq = periodic_hypotsq(rgrp, rprt, Bsize)
 *           = default code path at HEAD (after hotfix 5791098, 2024-04-08)
 *   early : sorted loop, EARLY_RETURN branch of prt_loop_inner (same hint)
 *   naive : every particle, Rsq = periodic_hypotsq(rgrp, rprt, Bsize) (NAIVE branch)
 *   truth : brute force over every particle, minimum-image distance in double
 * and the radial bin of Lee22's y_prof.cpp / ne_prof.cpp / dens_prof.cpp
 * (128 log bins, 0.03-2.5 R200c, bin 0 = everything inside 0.03 R200c).
 *
 * Usage: unit_harness key=value ...
 *   mode=targeted|population   label=<str>   out=<dir>   seed=<int>
 *   targeted:   n_per_class=<int> n_cloud=<int> n_background=<int>
 *   population: n_pop=<int> n_cloud_pop=<int>
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <tuple>
#include <random>
#include <string>
#include <algorithm>
#include <omp.h>

#include "fields.hpp"
#include "common_fields.hpp"
#include "callback.hpp"
#include "workspace.hpp"
#include "workspace_sorting.hpp"
#include "geom_utils.hpp"

namespace HarnessFields {
    FIELD(ParticleIDs, 1, uint64_t, FieldTypes::PrtFld, false);
}

using GrpF = GrpFields<IllustrisFields::GroupPos, IllustrisFields::Group_R_Crit200>;
using PrtF = PrtFields<IllustrisFields::Coordinates, HarnessFields::ParticleIDs>;
using AF   = AllFields<GrpF, PrtF>;

// Workspace<AF>::Sorting is a private member type; explicit instantiations are
// exempt from access checking, which lets us name it without touching the library.
namespace access_exemption {
    template<typename T> struct Box { using type = T; };
    struct Tag { };
    auto fetch (Tag);
    template<typename T> struct Thief { friend auto fetch (Tag) { return Box<T>{}; } };
    template struct Thief<grp_prt_detail::Workspace<AF>::Sorting>;
}
using Sorting = decltype(access_exemption::fetch(access_exemption::Tag{}))::type;
namespace GU = grp_prt_detail::GeomUtils;

// ---------------------------------------------------------------- constants
static const double Om = 0.3089, OL = 0.6911;   // TNG300 (Planck 2015)
static const double RHOC0 = 2.775e-8;           // 1e10 Msun/h / (kpc/h)^3, as in y_prof.cpp
static const double TOL_R = 0.05;               // ckpc/h: 3x the worst float rounding of a periodic difference
static const size_t NBIN = 128;                 // y_prof.cpp

static double r200c_comoving (double M10, double z)
{   // M10 in 1e10 Msun/h, returns comoving ckpc/h (TNG Group_R_Crit200 convention)
    const double E2 = Om * std::pow(1.0 + z, 3) + OL;
    return std::cbrt(3.0 * M10 / (4.0 * M_PI * 200.0 * RHOC0 * E2)) * (1.0 + z);
}

// ---------------------------------------------------------------- y_prof binning (verbatim formula)
struct YBins { coord_t logRmin, logRmax, dlogR; };
static YBins ybins (coord_t R200c)
{
    YBins b;
    b.logRmin = std::log(0.03F * R200c);
    b.logRmax = std::log(2.50F * R200c);
    b.dlogR = (b.logRmax - b.logRmin) / (coord_t)(NBIN - 1);
    return b;
}
static size_t ybin_float (const YBins &b, coord_t Rsq)
{   // NBIN means "not inserted"
    coord_t logR = (coord_t)0.5 * std::log(Rsq);
    if (logR > b.logRmax) return NBIN;
    size_t idx;
    if (logR < b.logRmin) idx = 0UL;
    else idx = 1UL + (size_t)((logR - b.logRmin) / b.dlogR);
    if (idx >= NBIN) return NBIN;
    return idx;
}
static size_t ybin_double (const YBins &b, double d2, double &edge_dist)
{
    const double lmin = b.logRmin, lmax = b.logRmax, dl = b.dlogR;
    const double logR = 0.5 * std::log(d2);
    const double u = (logR - lmin) / dl;
    double k = std::round(u);
    k = std::min(std::max(k, 0.0), (double)(NBIN - 1));
    edge_dist = std::fabs(logR - (lmin + k * dl));
    if (logR > lmax) return NBIN;
    if (logR < lmin) return 0;
    size_t idx = 1UL + (size_t)u;
    return idx >= NBIN ? NBIN : idx;
}

// ---------------------------------------------------------------- data types
enum { UNIFORM = 0, LOW_FACE, HIGH_FACE, LOW_EDGE, LOW_CORNER, HIGH_CORNER, MIXED,
       EXACT_ZERO, TOP, CELL_EDGE, POP, NCLS };
static const char *cls_name[NCLS] = { "uniform", "low_face", "high_face", "low_edge",
    "low_corner", "high_corner", "mixed_low_high", "exact_zero_coord", "coord_Bminus1ulp_or_B",
    "cell_edge_coords", "population_uniform" };

struct Grp {
    float pos[3];
    float R200c, R, Rsq;
    int cls;
    double M10, z;
};

enum { V_HINT = 0, V_PLAIN, V_EARLY, V_NAIVE, NVAR };
static const char *var_name[NVAR] = { "hint_sorted_c3fe9b8_default", "plain_sorted_HEAD_default",
                                      "early_return_hint", "naive_all_particles" };
enum { ACC_TRUE = 0, DROP_MISSED_GROSS, DROP_MISSED_PREC, DROP_DIST_GROSS, DROP_DIST_PREC,
       SPUR_GROSS, SPUR_PREC, DUP, BIN_GROSS, BIN_PREC, NCAT };
static const char *cat_name[NCAT] = { "accepted_true", "dropped_cell_not_searched_gross",
    "dropped_cell_not_searched_at_R", "dropped_by_distance_gross", "dropped_by_distance_at_R",
    "spurious_gross", "spurious_at_R", "double_counted", "wrong_bin_gross", "wrong_bin_at_edge" };

struct Stats {
    long n_groups = 0, n_true = 0, n_cand = 0, n_visits = 0, n_ranges = 0;
    long c[NVAR][NCAT] = {};
    // mechanism bookkeeping for the hint (periodic_to_add) values
    long visits_wrong_hint = 0, visits_wrong_p1m1 = 0, visits_wrong_other = 0;
    long hint_drop_gross_p1m1 = 0, hint_drop_gross_other = 0, hint_acc_with_wrong_hint = 0;
    long pred_lost = 0, pred_lost_hint_dropped = 0, hint_dropped_not_pred = 0;
    long range_hint_hist[3][3] = {};   // [axis][h+1]
    void add (const Stats &o)
    {
        n_groups += o.n_groups; n_true += o.n_true; n_cand += o.n_cand;
        n_visits += o.n_visits; n_ranges += o.n_ranges;
        for (int v = 0; v < NVAR; ++v) for (int k = 0; k < NCAT; ++k) c[v][k] += o.c[v][k];
        visits_wrong_hint += o.visits_wrong_hint; visits_wrong_p1m1 += o.visits_wrong_p1m1;
        visits_wrong_other += o.visits_wrong_other;
        hint_drop_gross_p1m1 += o.hint_drop_gross_p1m1; hint_drop_gross_other += o.hint_drop_gross_other;
        hint_acc_with_wrong_hint += o.hint_acc_with_wrong_hint;
        pred_lost += o.pred_lost; pred_lost_hint_dropped += o.pred_lost_hint_dropped;
        hint_dropped_not_pred += o.hint_dropped_not_pred;
        for (int a = 0; a < 3; ++a) for (int h = 0; h < 3; ++h) range_hint_hist[a][h] += o.range_hint_hist[a][h];
    }
};

struct Visit { uint64_t pid; float rsq_hint, rsq_plain, rsq_early; bool acc_early; int8_t h[3]; };
struct Cand  { uint64_t pid; double d2; float rsq_naive; int8_t hs[3]; };

struct GroupRow {
    long n_true = 0, n_acc[NVAR] = {}, n_pred_lost = 0, n_hint_drop_gross = 0, n_ranges = 0, n_visits = 0;
};

// per-pair outcome sink (population stacking, low-face cap test)
struct PairOut { float r; bool in_true, acc_hint, acc_plain, pred_lost; int8_t hs[3]; uint64_t pid; };

static int8_t hstar_of (double xp, double xg, double B)
{   // the integer k with |xp - xg + k B| <= B/2 (the hint the cell search should have produced)
    const double d = xp - xg;
    return d > 0.5 * B ? (int8_t)-1 : (d < -0.5 * B ? (int8_t)1 : (int8_t)0);
}

// ---------------------------------------------------------------- per-group evaluation
static Stats process_group (int gidx, const Grp &g, const Sorting &srt,
                            const float *sx, const uint64_t *sid,
                            const std::vector<float> &xyz, size_t N, float Bf,
                            GroupRow &row, std::vector<std::string> &disc, long disc_budget[NVAR][NCAT],
                            std::vector<PairOut> *pairs)
{
    Stats st; st.n_groups = 1;
    const double B = (double)Bf;
    const YBins yb = ybins(g.R200c);

    // ---- 1. the real cell search
    const auto ranges = srt.prt_idx_ranges(g.pos, g.R, g.Rsq);
    st.n_ranges = (long)ranges.size(); row.n_ranges = st.n_ranges;

    std::vector<Visit> vis;
    for (const auto &rg : ranges) {
        const auto &h = std::get<2>(rg);
        for (int a = 0; a < 3; ++a) st.range_hint_hist[a][h[a] + 1]++;
        for (size_t k = std::get<0>(rg); k != std::get<1>(rg); ++k) {
            const coord_t *rprt = sx + 3 * k;
            Visit v;
            v.pid = sid[k];
            v.rsq_hint  = GU::periodic_hypotsq(g.pos, rprt, Bf, h);       // c3fe9b8 default branch
            v.rsq_plain = GU::periodic_hypotsq(g.pos, rprt, Bf);          // HEAD default branch
            // EARLY_RETURN branch of prt_loop_inner, verbatim logic
            coord_t Rsq = (coord_t)0.0; bool rej = false;
            for (size_t ii = 0; ii != 3; ++ii) {
                coord_t dx = std::fabs(GU::periodic_dist_whint(g.pos[ii], rprt[ii], Bf, h[ii]));
                if (dx > g.R) { rej = true; break; }
                Rsq += dx * dx;
            }
            v.acc_early = !rej && !(Rsq > g.Rsq);
            v.rsq_early = rej ? -1.0F : Rsq;
            for (int a = 0; a < 3; ++a) v.h[a] = (int8_t)h[a];
            vis.push_back(v);
        }
    }
    st.n_visits = (long)vis.size(); row.n_visits = st.n_visits;
    std::sort(vis.begin(), vis.end(), [](const Visit &a, const Visit &b){ return a.pid < b.pid; });

    // ---- 2. brute-force truth over every particle (pid == original index)
    std::vector<Cand> cand;
    const double Rc = (double)g.R + 2.0 * TOL_R, Rc2 = Rc * Rc;
    const double gx = g.pos[0], gy = g.pos[1], gz = g.pos[2];
    for (size_t i = 0; i < N; ++i) {
        double dx = std::fabs((double)xyz[3*i] - gx);   if (dx > 0.5*B) dx = B - dx; if (dx > Rc) continue;
        double dy = std::fabs((double)xyz[3*i+1] - gy); if (dy > 0.5*B) dy = B - dy; if (dy > Rc) continue;
        double dz = std::fabs((double)xyz[3*i+2] - gz); if (dz > 0.5*B) dz = B - dz; if (dz > Rc) continue;
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > Rc2) continue;
        Cand c; c.pid = i; c.d2 = d2;
        c.rsq_naive = GU::periodic_hypotsq(g.pos, &xyz[3*i], Bf);       // NAIVE branch
        c.hs[0] = hstar_of(xyz[3*i], gx, B); c.hs[1] = hstar_of(xyz[3*i+1], gy, B); c.hs[2] = hstar_of(xyz[3*i+2], gz, B);
        cand.push_back(c);
    }
    st.n_cand = (long)cand.size();

    auto record = [&](int var, int cat, const Cand *c, const Visit *v, uint64_t pid, const char *note) {
        if (disc_budget[var][cat] <= 0) return;
        disc_budget[var][cat]--;
        char buf[1024];
        const float *p = &xyz[3*pid];
        std::snprintf(buf, sizeof buf,
            "%d,%s,%s,%s,%.6f,%.6f,%.6f,%.4f,%llu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%s\n",
            gidx, cls_name[g.cls], var_name[var], cat_name[cat], g.pos[0], g.pos[1], g.pos[2], g.R,
            (unsigned long long)pid, p[0], p[1], p[2],
            c ? std::sqrt(c->d2) : -1.0,
            v ? std::sqrt(std::max(0.0F, v->rsq_hint)) : -1.0, v ? std::sqrt(std::max(0.0F, v->rsq_plain)) : -1.0,
            v ? v->h[0] : 9, v ? v->h[1] : 9, v ? v->h[2] : 9,
            c ? c->hs[0] : (int)hstar_of(p[0], gx, B), c ? c->hs[1] : (int)hstar_of(p[1], gy, B),
            c ? c->hs[2] : (int)hstar_of(p[2], gz, B), note);
        disc.emplace_back(buf);
    };

    auto hint_wrong = [](const Visit &v, const int8_t hs[3], bool &p1m1_only) {
        bool any = false, other = false;
        for (int a = 0; a < 3; ++a) if (v.h[a] != hs[a]) {
            any = true;
            if (!(v.h[a] == 1 && hs[a] == -1)) other = true;
        }
        p1m1_only = any && !other;
        return any;
    };

    // ---- 3. merge visits and truth candidates by particle id
    size_t iv = 0, ic = 0;
    while (iv < vis.size() || ic < cand.size()) {
        uint64_t pid;
        if (ic < cand.size() && (iv >= vis.size() || cand[ic].pid <= vis[iv].pid)) pid = cand[ic].pid;
        else pid = vis[iv].pid;
        const Cand *c = (ic < cand.size() && cand[ic].pid == pid) ? &cand[ic] : nullptr;
        size_t v0 = iv; while (iv < vis.size() && vis[iv].pid == pid) ++iv;
        const size_t nvis = iv - v0;
        if (c) ++ic;

        int8_t hs[3];
        if (c) { for (int a = 0; a < 3; ++a) hs[a] = c->hs[a]; }
        else {
            const float *p = &xyz[3*pid];
            hs[0] = hstar_of(p[0], gx, B); hs[1] = hstar_of(p[1], gy, B); hs[2] = hstar_of(p[2], gz, B);
        }

        // acceptance counts
        long nacc[NVAR] = {0, 0, 0, 0};
        const Visit *first_acc[NVAR] = {nullptr, nullptr, nullptr, nullptr};
        bool any_wrong = false, all_wrong_p1m1 = true;
        for (size_t k = v0; k < iv; ++k) {
            const Visit &v = vis[k];
            if (!(v.rsq_hint > g.Rsq))  { nacc[V_HINT]++;  if (!first_acc[V_HINT])  first_acc[V_HINT]  = &v; }
            if (!(v.rsq_plain > g.Rsq)) { nacc[V_PLAIN]++; if (!first_acc[V_PLAIN]) first_acc[V_PLAIN] = &v; }
            if (v.acc_early)            { nacc[V_EARLY]++; if (!first_acc[V_EARLY]) first_acc[V_EARLY] = &v; }
            bool p1m1;
            if (hint_wrong(v, hs, p1m1)) {
                st.visits_wrong_hint++; any_wrong = true;
                if (p1m1) st.visits_wrong_p1m1++; else { st.visits_wrong_other++; all_wrong_p1m1 = false; }
                if (!(v.rsq_hint > g.Rsq)) st.hint_acc_with_wrong_hint++;
            }
        }
        if (c && !(c->rsq_naive > g.Rsq)) nacc[V_NAIVE] = 1;

        const bool in_true = c && c->d2 <= (double)g.Rsq;
        const bool near_R  = c && std::fabs(std::sqrt(c->d2) - (double)g.R) < TOL_R;
        const bool pred    = in_true && (hs[0] == -1 || hs[1] == -1 || hs[2] == -1);

        if (in_true) {
            st.n_true++; row.n_true++;
            if (pred) { st.pred_lost++; row.n_pred_lost++; }
            for (int var = 0; var < NVAR; ++var) {
                const bool sorted = var != V_NAIVE;
                if (sorted && nvis == 0) {
                    const int cat = near_R ? DROP_MISSED_PREC : DROP_MISSED_GROSS;
                    st.c[var][cat]++; record(var, cat, c, nullptr, pid, "true pair in no searched cell");
                    continue;
                }
                if (nacc[var] == 0) {
                    const int cat = near_R ? DROP_DIST_PREC : DROP_DIST_GROSS;
                    st.c[var][cat]++;
                    record(var, cat, c, nvis ? &vis[v0] : nullptr, pid,
                           (var == V_HINT && any_wrong) ? (all_wrong_p1m1 ? "hint +1 where -1 needed" : "other wrong hint") : "");
                    if (var == V_HINT) {
                        if (cat == DROP_DIST_GROSS) { if (any_wrong && all_wrong_p1m1) st.hint_drop_gross_p1m1++; else st.hint_drop_gross_other++; row.n_hint_drop_gross++; }
                        if (pred) st.pred_lost_hint_dropped++; else st.hint_dropped_not_pred++;
                    }
                    continue;
                }
                st.c[var][ACC_TRUE]++; row.n_acc[var]++;
                if (nacc[var] > 1) { st.c[var][DUP]++; record(var, DUP, c, first_acc[var], pid, "accepted more than once"); }
                // radial bin
                coord_t rsq_v = var == V_HINT ? first_acc[var]->rsq_hint : var == V_PLAIN ? first_acc[var]->rsq_plain
                              : var == V_EARLY ? first_acc[var]->rsq_early : c->rsq_naive;
                double edge;
                const size_t bt = ybin_double(yb, c->d2, edge);
                const size_t bv = ybin_float(yb, rsq_v);
                if (bt != bv) {
                    const double tol = TOL_R / std::sqrt(std::max(c->d2, 1e-30)) + 1e-5;
                    const int cat = edge < tol ? BIN_PREC : BIN_GROSS;
                    st.c[var][cat]++; record(var, cat, c, var == V_NAIVE ? nullptr : first_acc[var], pid, "bin differs");
                }
            }
        } else {
            for (int var = 0; var < NVAR; ++var) if (nacc[var] > 0) {
                const int cat = near_R ? SPUR_PREC : SPUR_GROSS;
                st.c[var][cat]++; record(var, cat, c, nvis ? &vis[v0] : nullptr, pid, "accepted outside R");
                if (nacc[var] > 1) st.c[var][DUP]++;
            }
        }
        if (pairs && in_true) {
            PairOut po; po.r = (float)std::sqrt(c->d2); po.in_true = true;
            po.acc_hint = nacc[V_HINT] > 0; po.acc_plain = nacc[V_PLAIN] > 0; po.pred_lost = pred;
            for (int a = 0; a < 3; ++a) po.hs[a] = hs[a];
            po.pid = pid;
            pairs->push_back(po);
        }
    }
    return st;
}

// ---------------------------------------------------------------- main
static std::string arg (int argc, char **argv, const char *key, const char *def)
{
    const size_t n = std::strlen(key);
    for (int i = 1; i < argc; ++i)
        if (!std::strncmp(argv[i], key, n) && argv[i][n] == '=') return std::string(argv[i] + n + 1);
    return def;
}

int main (int argc, char **argv)
{
    const std::string mode  = arg(argc, argv, "mode", "targeted");
    const std::string label = arg(argc, argv, "label", "run");
    const std::string out   = arg(argc, argv, "out", ".");
    const uint64_t seed     = std::stoull(arg(argc, argv, "seed", "20261007"));
    const int n_per_class   = std::stoi(arg(argc, argv, "n_per_class", "150"));
    const int n_cloud       = std::stoi(arg(argc, argv, "n_cloud", "2500"));
    const long n_background = std::stol(arg(argc, argv, "n_background", "1000000"));
    const int n_pop         = std::stoi(arg(argc, argv, "n_pop", "12000"));
    const int n_cloud_pop   = std::stoi(arg(argc, argv, "n_cloud_pop", "300"));

    const float Bf = 205000.0F;               // TNG300-1 BoxSize, ckpc/h
    const double B = (double)Bf;
    const float acell = Bf / 64UL;            // identical expression to Sorting::acell
    const float B_1ulp = std::nextafter(Bf, 0.0F);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    auto draw_axis = [&]{ return (int)(U01(rng) * 3.0) % 3; };

    std::vector<Grp> grps;
    auto make_grp = [&](int cls, double M10, double z) {
        Grp g; g.cls = cls; g.M10 = M10; g.z = z;
        g.R200c = (float)r200c_comoving(M10, z);
        g.R = 2.5F * g.R200c;                 // radius::Simple: scaling (float 2.5) * Group_R_Crit200
        g.Rsq = g.R * g.R;                    // grp_loop: grp_radii_sq = grp_radii * grp_radii
        return g;
    };

    if (mode == "targeted") {
        auto lowx  = [&](float R){ return (float)(U01(rng) * R); };
        auto highx = [&](float R){ return (float)(B - U01(rng) * R); };
        auto intx  = [&](float R){ const double lo = R + 2.0 * acell; return (float)(lo + U01(rng) * (B - 2.0 * lo)); };
        const float cellvals[] = { 0.0F, acell, std::nextafter(acell, 0.0F), std::nextafter(acell, Bf),
                                   2.0F * acell, 62.0F * acell, 63.0F * acell, std::nextafter(63.0F * acell, 0.0F),
                                   std::nextafter(63.0F * acell, Bf), B_1ulp };
        for (int cls = 0; cls < POP; ++cls)
            for (int i = 0; i < n_per_class; ++i) {
                Grp g = make_grp(cls, std::pow(10.0, 3.0 + 2.0 * U01(rng)), 2.0 * U01(rng));
                const float R = g.R;
                int a = draw_axis(), b = (a + 1 + (int)(U01(rng) * 2.0) % 2) % 3, c3 = 3 - a - b;
                switch (cls) {
                case UNIFORM:    for (int k = 0; k < 3; ++k) g.pos[k] = (float)(U01(rng) * B); break;
                case LOW_FACE:   g.pos[a] = lowx(R);  g.pos[b] = intx(R); g.pos[c3] = intx(R); break;
                case HIGH_FACE:  g.pos[a] = highx(R); g.pos[b] = intx(R); g.pos[c3] = intx(R); break;
                case LOW_EDGE:   g.pos[a] = lowx(R);  g.pos[b] = lowx(R); g.pos[c3] = intx(R); break;
                case LOW_CORNER: for (int k = 0; k < 3; ++k) g.pos[k] = lowx(R); break;
                case HIGH_CORNER:for (int k = 0; k < 3; ++k) g.pos[k] = highx(R); break;
                case MIXED: {
                    g.pos[a] = lowx(R); g.pos[b] = highx(R);
                    const double u = U01(rng); g.pos[c3] = u < 1.0/3 ? lowx(R) : (u < 2.0/3 ? highx(R) : intx(R)); break; }
                case EXACT_ZERO:
                    g.pos[a] = 0.0F; g.pos[b] = U01(rng) < 0.5 ? 0.0F : lowx(R); g.pos[c3] = U01(rng) < 0.5 ? intx(R) : highx(R); break;
                case TOP:
                    g.pos[a] = U01(rng) < 0.5 ? B_1ulp : Bf;
                    { const double u = U01(rng); g.pos[b] = u < 1.0/3 ? lowx(R) : (u < 2.0/3 ? highx(R) : intx(R)); }
                    g.pos[c3] = intx(R); break;
                case CELL_EDGE:
                    for (int k = 0; k < 3; ++k) g.pos[k] = cellvals[(int)(U01(rng) * 10.0) % 10];
                    break;
                }
                grps.push_back(g);
            }
    } else if (mode == "population") {
        // uniform positions, dn/dlnM ~ M^-1 over Lee22's mass range 1e13 - 10^14.2 Msun/h,
        // redshift from five full snapshots z = 0, 0.5, 1, 1.5, 2
        const double zs[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
        const double lo = 1e3, hi = std::pow(10.0, 4.2);
        for (int i = 0; i < n_pop; ++i) {
            const double u = U01(rng);
            const double M10 = 1.0 / (1.0 / lo - u * (1.0 / lo - 1.0 / hi));   // p(lnM) ~ M^-1
            Grp g = make_grp(POP, M10, zs[(int)(U01(rng) * 5.0) % 5]);
            for (int k = 0; k < 3; ++k) g.pos[k] = (float)(U01(rng) * B);
            grps.push_back(g);
        }
    } else { std::fprintf(stderr, "unknown mode\n"); return 1; }

    // ---- particles (float coordinates as after convert_coords double -> float)
    std::vector<float> xyz;
    auto push = [&](double x, double y, double z) {
        auto wrap = [&](double v){ v -= B * std::floor(v / B); return (float)v; };   // (float) may give exactly B
        xyz.push_back(wrap(x)); xyz.push_back(wrap(y)); xyz.push_back(wrap(z));
    };
    auto ball = [&](const Grp &g, double rad, int n, bool snap_faces) {
        std::normal_distribution<double> G(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            double v[3] = {G(rng), G(rng), G(rng)};
            const double nv = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            const double r = rad * std::cbrt(U01(rng));
            float q[3];
            for (int k = 0; k < 3; ++k) {
                double w = g.pos[k] + r * v[k] / nv;
                w -= B * std::floor(w / B);
                q[k] = (float)w;                      // may round to exactly B, as TNG doubles near B do
            }
            if (snap_faces && i % 50 == 0) {
                // put every 50th cloud particle exactly on the face plane (coordinate 0.0f or the float value B)
                const int k = (int)(U01(rng) * 3.0) % 3;
                if (g.pos[k] < g.R || Bf - g.pos[k] < g.R) q[k] = (i % 100 == 0) ? 0.0F : Bf;
            }
            xyz.push_back(q[0]); xyz.push_back(q[1]); xyz.push_back(q[2]);
        }
    };
    if (mode == "targeted") {
        for (long i = 0; i < n_background; ++i) push(U01(rng) * B, U01(rng) * B, U01(rng) * B);
        const float special[] = { 0.0F, 1e-3F, std::nextafter(acell, 0.0F), acell, std::nextafter(acell, Bf),
                                  std::nextafter(63.0F * acell, 0.0F), 63.0F * acell, std::nextafter(63.0F * acell, Bf),
                                  B_1ulp, Bf };
        for (float x : special) for (float y : special) for (float z : special) {
            xyz.push_back(x); xyz.push_back(y); xyz.push_back(z);
        }
        for (const auto &g : grps) ball(g, 1.02 * g.R, n_cloud, true);
    } else {
        for (const auto &g : grps) ball(g, (double)g.R, n_cloud_pop, false);
    }
    const size_t N = xyz.size() / 3;
    std::vector<uint64_t> ids(N); std::iota(ids.begin(), ids.end(), 0ULL);
    long n_at_B = 0, n_at_0 = 0;
    for (size_t i = 0; i < 3 * N; ++i) { if (xyz[i] == Bf) ++n_at_B; if (xyz[i] == 0.0F) ++n_at_0; }
    std::fprintf(stderr, "[%s] mode=%s groups=%zu particles=%zu coords==B:%ld coords==0:%ld acell=%.6f\n",
                 label.c_str(), mode.c_str(), grps.size(), N, n_at_B, n_at_0, acell);

    // ---- the real Sorting (cell lists + per-cell offsets)
    void *props[2] = { (void *)xyz.data(), (void *)ids.data() };
    Sorting srt(N, Bf, props);
    const float *sx = (const float *)srt.tmp_prt_properties_sorted[0];
    const uint64_t *sid = (const uint64_t *)srt.tmp_prt_properties_sorted[1];

    // ---- probe: which hints does the real prt_idx_ranges attach to wrapped cells?
    {
        FILE *f = std::fopen((out + "/" + label + "_" + mode + "_probe_hints.csv").c_str(), "w");
        std::fprintf(f, "probe,grp_x,grp_y,grp_z,R,cell_x,cell_y,cell_z,hint_x,hint_y,hint_z,n_prt\n");
        const float probes[4][3] = { {0.3F * acell, 32.5F * acell, 32.5F * acell},     // near low x face
                                     {63.7F * acell, 32.5F * acell, 32.5F * acell},    // near high x face
                                     {0.3F * acell, 0.3F * acell, 0.3F * acell},       // low corner
                                     {63.7F * acell, 63.7F * acell, 63.7F * acell} };  // high corner
        const char *pn[4] = {"low_x_face", "high_x_face", "low_corner", "high_corner"};
        for (int p = 0; p < 4; ++p) {
            const float R = 0.9F * acell, Rsq = R * R;
            for (const auto &rg : srt.prt_idx_ranges(probes[p], R, Rsq)) {
                const float *q = sx + 3 * std::get<0>(rg);
                const auto &h = std::get<2>(rg);
                std::fprintf(f, "%s,%.3f,%.3f,%.3f,%.3f,%zu,%zu,%zu,%d,%d,%d,%zu\n", pn[p], probes[p][0], probes[p][1], probes[p][2], R,
                             std::min((size_t)(q[0] / acell), (size_t)63), std::min((size_t)(q[1] / acell), (size_t)63),
                             std::min((size_t)(q[2] / acell), (size_t)63), h[0], h[1], h[2],
                             std::get<1>(rg) - std::get<0>(rg));
            }
        }
        std::fclose(f);
    }

    // ---- evaluate all groups
    std::vector<Stats> cls_stats(NCLS);
    std::vector<GroupRow> rows(grps.size());
    std::vector<std::string> disc_all;
    std::vector<PairOut> lowface_pairs_all;            // targeted: single-low-face groups
    std::vector<int> lowface_group_of_pair;
    const int NX = 125;                                // population: x = r / R200c bins of 0.02
    std::vector<long> pop_true(NX, 0), pop_hint(NX, 0), pop_plain(NX, 0), pop_pred(NX, 0);
    std::vector<double> pop_true_xR(NX, 0.0);          // sum over true pairs of r (ckpc/h), for the analytic law

    #pragma omp parallel
    {
        std::vector<Stats> loc(NCLS);
        std::vector<std::string> disc;
        long budget[NCLS][NVAR][NCAT];
        for (int a = 0; a < NCLS; ++a) for (int v = 0; v < NVAR; ++v) for (int k = 0; k < NCAT; ++k)
            budget[a][v][k] = (k == ACC_TRUE) ? 0 : 3;   // per thread; merged list is trimmed later
        std::vector<PairOut> lf_pairs; std::vector<int> lf_gid;
        std::vector<long> pt(NX, 0), ph(NX, 0), pp(NX, 0), pr(NX, 0); std::vector<double> pxr(NX, 0.0);
        std::vector<PairOut> pairs;

        #pragma omp for schedule(dynamic, 2)
        for (size_t gi = 0; gi < grps.size(); ++gi) {
            const Grp &g = grps[gi];
            pairs.clear();
            int nlow = 0; for (int k = 0; k < 3; ++k) nlow += g.pos[k] < g.R;
            const bool want_pairs = (mode == "population") || (g.cls == LOW_FACE && nlow == 1);
            Stats s = process_group((int)gi, g, srt, sx, sid, xyz, N, Bf, rows[gi], disc, budget[g.cls],
                                    want_pairs ? &pairs : nullptr);
            loc[g.cls].add(s);
            if (mode == "population") {
                for (const auto &p : pairs) {
                    const int xb = (int)(p.r / g.R200c / 0.02F);
                    if (xb < 0 || xb >= NX) continue;
                    pt[xb]++; ph[xb] += p.acc_hint; pp[xb] += p.acc_plain; pr[xb] += p.pred_lost; pxr[xb] += p.r;
                }
            } else if (want_pairs) {
                for (const auto &p : pairs) { lf_pairs.push_back(p); lf_gid.push_back((int)gi); }
            }
        }
        #pragma omp critical
        {
            for (int a = 0; a < NCLS; ++a) cls_stats[a].add(loc[a]);
            disc_all.insert(disc_all.end(), disc.begin(), disc.end());
            lowface_pairs_all.insert(lowface_pairs_all.end(), lf_pairs.begin(), lf_pairs.end());
            lowface_group_of_pair.insert(lowface_group_of_pair.end(), lf_gid.begin(), lf_gid.end());
            for (int x = 0; x < NX; ++x) { pop_true[x] += pt[x]; pop_hint[x] += ph[x]; pop_plain[x] += pp[x]; pop_pred[x] += pr[x]; pop_true_xR[x] += pxr[x]; }
        }
    }

    // ---- outputs
    const std::string stem = out + "/" + label + "_" + mode;
    {   // summary (JSON)
        FILE *f = std::fopen((stem + "_summary.json").c_str(), "w");
        std::fprintf(f, "{\n  \"label\": \"%s\", \"mode\": \"%s\", \"seed\": %llu, \"Bsize\": %.1f, \"acell\": %.6f,\n"
                        "  \"n_groups\": %zu, \"n_particles\": %zu, \"particle_coords_equal_B\": %ld, \"particle_coords_equal_0\": %ld,\n"
                        "  \"tol_R_ckpc_h\": %.3f, \"n_background\": %ld, \"n_special_lattice\": %d, \"n_cloud\": %d,\n  \"classes\": {\n",
                     label.c_str(), mode.c_str(), (unsigned long long)seed, B, acell, grps.size(), N, n_at_B, n_at_0, TOL_R,
                     mode == "targeted" ? n_background : 0L, mode == "targeted" ? 1000 : 0, mode == "targeted" ? n_cloud : n_cloud_pop);
        Stats tot;
        bool first = true;
        for (int a = 0; a <= NCLS; ++a) {
            const Stats &s = a < NCLS ? cls_stats[a] : tot;
            if (a < NCLS) { if (s.n_groups == 0) continue; tot.add(s); }
            std::fprintf(f, "%s    \"%s\": {\"n_groups\": %ld, \"n_true_pairs\": %ld, \"n_visits\": %ld, \"n_ranges\": %ld,\n",
                         first ? "" : ",\n", a < NCLS ? cls_name[a] : "ALL", s.n_groups, s.n_true, s.n_visits, s.n_ranges);
            first = false;
            for (int v = 0; v < NVAR; ++v) {
                std::fprintf(f, "      \"%s\": {", var_name[v]);
                for (int k = 0; k < NCAT; ++k) std::fprintf(f, "%s\"%s\": %ld", k ? ", " : "", cat_name[k], s.c[v][k]);
                std::fprintf(f, "},\n");
            }
            std::fprintf(f, "      \"hint_mechanism\": {\"visits_with_wrong_hint\": %ld, \"wrong_hint_all_plus1_where_minus1_needed\": %ld,"
                            " \"wrong_hint_other_pattern\": %ld, \"hint_dropped_gross_plus1_for_minus1\": %ld, \"hint_dropped_gross_other\": %ld,"
                            " \"hint_accepted_despite_wrong_hint\": %ld, \"true_pairs_reached_across_low_face\": %ld,"
                            " \"of_which_hint_dropped\": %ld, \"hint_dropped_not_across_low_face\": %ld,"
                            " \"range_hint_counts_axis_x_minus1_0_plus1\": [%ld, %ld, %ld], \"range_hint_counts_axis_y\": [%ld, %ld, %ld],"
                            " \"range_hint_counts_axis_z\": [%ld, %ld, %ld]}}",
                         s.visits_wrong_hint, s.visits_wrong_p1m1, s.visits_wrong_other, s.hint_drop_gross_p1m1,
                         s.hint_drop_gross_other, s.hint_acc_with_wrong_hint, s.pred_lost, s.pred_lost_hint_dropped,
                         s.hint_dropped_not_pred,
                         s.range_hint_hist[0][0], s.range_hint_hist[0][1], s.range_hint_hist[0][2],
                         s.range_hint_hist[1][0], s.range_hint_hist[1][1], s.range_hint_hist[1][2],
                         s.range_hint_hist[2][0], s.range_hint_hist[2][1], s.range_hint_hist[2][2]);
        }
        std::fprintf(f, "\n  }\n}\n");
        std::fclose(f);
    }
    {   // per-group table
        FILE *f = std::fopen((stem + "_groups.csv").c_str(), "w");
        std::fprintf(f, "gidx,cls,x,y,z,M200c_1e10Msun_h,z_red,R200c_ckpc_h,R_ckpc_h,min_dist_low_face,min_dist_high_face,n_low_faces_within_R,"
                        "n_true,n_acc_hint,n_acc_plain,n_acc_early,n_acc_naive,n_true_across_low_face,n_hint_dropped_gross,n_ranges,n_visits\n");
        for (size_t gi = 0; gi < grps.size(); ++gi) {
            const Grp &g = grps[gi]; const GroupRow &r = rows[gi];
            float dlow = std::min({g.pos[0], g.pos[1], g.pos[2]});
            float dhigh = std::min({Bf - g.pos[0], Bf - g.pos[1], Bf - g.pos[2]});
            int nlow = 0; for (int k = 0; k < 3; ++k) nlow += g.pos[k] < g.R;
            std::fprintf(f, "%zu,%s,%.6f,%.6f,%.6f,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld\n",
                         gi, cls_name[g.cls], g.pos[0], g.pos[1], g.pos[2], g.M10, g.z, g.R200c, g.R, dlow, dhigh, nlow,
                         r.n_true, r.n_acc[V_HINT], r.n_acc[V_PLAIN], r.n_acc[V_EARLY], r.n_acc[V_NAIVE], r.n_pred_lost,
                         r.n_hint_drop_gross, r.n_ranges, r.n_visits);
        }
        std::fclose(f);
    }
    {   // discrepancy examples (up to 3 per thread x class x variant x category)
        FILE *f = std::fopen((stem + "_discrepancy_examples.csv").c_str(), "w");
        std::fprintf(f, "gidx,cls,variant,category,grp_x,grp_y,grp_z,R,pid,prt_x,prt_y,prt_z,d_true,d_hint,d_plain,hint_x,hint_y,hint_z,"
                        "needed_hint_x,needed_hint_y,needed_hint_z,note\n");
        std::sort(disc_all.begin(), disc_all.end());
        for (const auto &s : disc_all) std::fputs(s.c_str(), f);
        std::fclose(f);
    }
    if (mode == "targeted") {   // single-low-face groups: per-pair r, distance to the face, outcome
        FILE *f = std::fopen((stem + "_lowface_pairs.csv").c_str(), "w");
        std::fprintf(f, "gidx,pid,r,d_face,R,acc_hint,acc_plain,across_low_face\n");
        for (size_t i = 0; i < lowface_pairs_all.size(); ++i) {
            const auto &p = lowface_pairs_all[i]; const Grp &g = grps[lowface_group_of_pair[i]];
            int ax = 0; for (int k = 0; k < 3; ++k) if (g.pos[k] < g.R) ax = k;
            std::fprintf(f, "%d,%llu,%.4f,%.4f,%.4f,%d,%d,%d\n", lowface_group_of_pair[i], (unsigned long long)p.pid, p.r, g.pos[ax], g.R,
                         (int)p.acc_hint, (int)p.acc_plain, (int)(p.hs[ax] == -1));
        }
        std::fclose(f);
    } else {
        FILE *f = std::fopen((stem + "_stack_vs_x.csv").c_str(), "w");
        std::fprintf(f, "x_lo,x_hi,n_true,n_acc_hint,n_acc_plain,n_true_across_low_face,sum_r_true_ckpc_h\n");
        for (int x = 0; x < NX; ++x)
            std::fprintf(f, "%.2f,%.2f,%ld,%ld,%ld,%ld,%.6e\n", 0.02 * x, 0.02 * (x + 1), pop_true[x], pop_hint[x], pop_plain[x], pop_pred[x], pop_true_xR[x]);
        std::fclose(f);
    }
    std::fprintf(stderr, "[%s] done, outputs at %s_*\n", label.c_str(), stem.c_str());
    return 0;
}
