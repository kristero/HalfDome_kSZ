# Fitting FLAMINGO profiles for HalfDome without double counting (2026-10-13)

**Question (user):** inside R200c the plan is to fit everything FLAMINGO has. Beyond R200c, FLAMINGO has satellites and
subhaloes that HalfDome does not have, especially around low-mass haloes. Around high-mass haloes, HalfDome does have
resolved haloes beyond R200c. How should the 2-3 R200c profiles be fitted so that nothing HalfDome already paints is
counted twice?

**Short answer.**
- **Inside R200c:** fit everything, as planned.
- **Beyond R200c, sort each FLAMINGO structure by one test:** would HalfDome paint it as a separate halo? That depends on
  the structure, not on the host's mass.
- **What HalfDome paints separately:** exactly what FLAMINGO calls separate FoF groups (other centrals) above the
  painting threshold. The counts agree at every mass.
- **What it doesn't:** FLAMINGO's own satellites are merged into the host by HalfDome's halo finder too, even beyond
  R200c, where they far outnumber the separate haloes. They must stay in the host's profile.
- **The recipe:** keep the host's own FoF group (central and all its satellites), and keep all groups below the
  threshold. Give gas bound to other painted groups, and unbound gas lying closer to them, to those groups. This is the
  A2-ii scaled-distance partition of Test A.
- **Where it matters:** inside 2 R200c it changes nothing (≤ 0.4% of any band). At 3-4 R200c it moves 18-46% of the
  band to painted neighbours, at low host masses too.

## 1. Inside R200c: include everything

- **Who owns the gas.** In FLAMINGO, 91-95% of the y and the electrons inside R200c are bound to the host central, 4-9%
  are unbound, and 0.1-1.3% are bound to the host's own satellites (Tables 2-3). No gas there belongs to another halo
  that HalfDome paints.
- **HalfDome.** It never has a separate halo inside R200c: 0.00 per host in every mass bin at z < 0.08 (Table 1), and 1
  of 1,143 hosts of ~1e15 Msun at z < 0.35.
- **So fitting everything inside R200c paints exactly what HalfDome needs.** The one trap is selection, found in
  Test A: haloes with a massive satellite (merging systems) look under-pressured in M200c units. HalfDome's catalogue
  contains them too, so calibrate on all haloes, not on "clean" ones.

## 2. Beyond R200c: which structures HalfDome paints separately

Table 1 counts the other haloes above the 6.6e12 Msun painting threshold around each host, in 3-D shells. FLAMINGO
haloes are split into separate FoF groups (other centrals) and the host's own satellites (subhaloes of the same FoF
group, bound mass >= 6.6e12).

- **1-2 R200c.** HalfDome has 0.00-0.08 separate haloes per host up to 10^15.1 Msun, and FLAMINGO has 0.00-0.09 other
  centrals. FLAMINGO's own satellites number 0.07 (10^13.6) to 1.7 (10^15.0) per host there, and 4.0 above 10^15.1, so
  10 to 30 times more. HalfDome's halo finder therefore merges these satellites into the host, as FLAMINGO's FoF does.
- **2-4 R200c.** HalfDome has 0.12-3.0 separate haloes per host (z < 0.08) against 0.13-2.8 FLAMINGO other centrals:
  ratios of 0.9-1.5. The excess is what two known effects predict:
  - HalfDome has 16-32% more haloes of 0.7-3e13 Msun per comoving volume than FLAMINGO at z = 0 (the typical neighbour
    mass);
  - at z ~ 0.065 a 4 R200c sphere encloses about 12% more comoving volume than at z = 0.
  If HalfDome painted FLAMINGO's satellites separately, it would have up to 0.1-1.8 more per host; it does not.
- **"Resolved haloes beyond R200c count as subhaloes?"** No. HalfDome's separate haloes around massive hosts (Figure 1)
  are FLAMINGO's separate FoF groups. They sit mostly beyond 2 R200c and there are more around massive hosts, simply
  because the 4 R200c sphere of a cluster encloses more volume.
- **Redshift.** At fixed mass the count grows with redshift (HalfDome z ~ 0.19 vs z < 0.08: +10-50%), because R200c
  corresponds to a larger comoving volume at higher z. The bookkeeping must be repeated at every calibration snapshot.

![Neighbour counts](figures/neighbour_counts.png)

## 3. Gas ownership and the size of the double counting

Tables 2-3 split each band's y and electrons among the host's own FoF group (central + own satellites), unbound gas,
groups below the threshold (absent from HalfDome) and groups above it (painted by HalfDome). The last column is what
the A2-ii partition gives to painted neighbours: their bound gas plus unbound gas closer to them in r/R200c.

- **1-2 R200c:** other painted groups hold 0.0-0.1% of the y and electrons, and A2-ii moves ≤ 0.4%. A fit to 2 R200c
  is the same with or without the partition.
- **2-3 R200c:**
  - other painted groups hold 1-4%, and A2-ii moves 7-10% (y) and 4-13% (electrons);
  - own satellites hold 3-14% of the y here (more around massive hosts). Keep this gas: HalfDome does not paint those
    satellites, so the host profile is the only place it can go.
- **3-4 R200c:** other painted groups hold 9-23% of the y and 4-13% of the electrons, and A2-ii moves 38-46% (y) and
  18-35% (electrons).
- **Low-mass hosts are not exempt.** A 10^13.6 Msun host has only 0.12 painted haloes within 4 R200c, but its 4 R200c
  sphere overlaps those of 0.7 painted haloes on average, a quarter of them heavier than the host. They own 9% of the y
  in its 3-4 R200c band, and A2-ii gives them 38%.
- **Integrated (Test A):** A2-ii removes 2.4-7.4% of Y(<4R) and 5.1-7.6% of N_e(<4R). Fitting all gas to 4 R200c and
  painting every HalfDome halo would therefore over-paint:
  - tSZ by +1.5% at ell = 100 (+0.3% at ell = 1000);
  - the kSZ 1-halo term by +2.7% at ell = 1000;
  - the mean DM by +3.4% and the DM 16th percentile by +6.2%.

![Gas ownership](figures/gas_ownership.png)

## 4. Recipe

1. **Inside R200c:** all gas (recently AGN-heated gas removed), all haloes.
2. **Ownership beyond R200c (A2-ii):**
   - every gas particle goes to the painted halo with the smallest r/R200c;
   - painted haloes are FLAMINGO centrals with M200c above the threshold that has HalfDome's abundance, about
     5.7e12 Msun for HalfDome's 6.6e12 (the current partition used 6.6e12, which slightly under-removes);
   - the host's own satellites stay with the host, and groups below the threshold stay with whichever painted halo is
     closest in scaled distance.
   - Each particle is then painted exactly once, on average over the population.
3. **Fit and cut at 4 R200c**, with the two-component profile (B12/B16 plus an outer power law; Test F), on the
   A2-ii-owned profiles. A fit to 2 R200c needs no partition but misses the 2-4 R200c gas (DM -22% with a 2 R200c cut).
4. **Redo the partition at each calibration redshift.**
5. **Validate with a closure test:** paint FLAMINGO's own halo catalogue with the calibrated profiles and compare with
   its lightcone y and DM maps. This is the only direct check that nothing is missed or counted twice.

## 5. Caveats

- **Mass threshold.** HalfDome's halo abundance near the threshold differs from FLAMINGO's: different cosmology,
  abundance matching to Tinker08, and baryonic mass loss in FLAMINGO. Matching by abundance sets FLAMINGO's threshold at
  ~5.7e12 Msun. Changing the partition threshold needs the particle-level partition, i.e. a new particle pass.
- **Small samples.** HalfDome has few hosts above 10^14.7 Msun at z < 0.08 (62, 29 and 8 per bin), so the 2-4 R200c
  comparison is noisy there. The 1-2 R200c result does not depend on those bins.
- **Satellite masses.** FLAMINGO satellite masses are HBT bound masses after stripping; HalfDome masses are M200c of FoF
  groups, abundance-matched.
- **Partition holds on average only.** The removal depends on environment, and HalfDome paints one spherical mean
  profile per mass. Total gas is therefore conserved on average over the population, not halo by halo.

## Files

- `flamingo_neighbour_query.py` (read-only SOAP query on maas, 33 s) -> `flamingo_neighbours.npz`
- `halfdome_neighbour_census.py` -> `halfdome_neighbours.npz` (z = 0.02-0.25), `halfdome_neighbours_lowz.npz`
  (z = 0.005-0.08)
- `compare_neighbours.py` -> `neighbour_counts.json`; `gas_bookkeeping.py` -> `gas_bookkeeping.json` (from the
  particle-pass owner and partition accumulators; no new particle reads)
- `make_figures.py` -> `figures/`; `report_tables.md` holds the generated Tables 1-3; `report_numbers.json`

## Tables

See `report_tables.md` (generated from the json files):
- Table 1: neighbour counts;
- Tables 2-3: ownership shares for 10^13.5-13.7, 10^14.1-14.3, 10^14.5-14.7 and >= 10^14.9 Msun hosts.
