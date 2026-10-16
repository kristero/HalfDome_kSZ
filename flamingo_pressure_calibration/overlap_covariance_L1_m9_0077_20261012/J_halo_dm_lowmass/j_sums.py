"""Test J1's checksum manifest SHA256SUMS (local; rewrites only SHA256SUMS and its log post/sums_update_j1.json).

  python -B j_sums.py     re-hash every file of J1's manifest and add the files of J1's 2026-10-16 revision (the Mecke
                          whole-atom run results_mecke/ without work/, its local smoke check results_mecke_smoke/ without
                          work/, j_mecke_atom.py, j_mecke_update.py, j_provenance.py, j_sums.py, the post/ records and
                          the archive of the superseded half-atom sens products); every changed, added or removed line is
                          logged with its old and new digest.  Run it last for J1, after the final verify_j.py (and
                          before j2_sums.py j2, which hashes J2's files).
Format: '<sha256>  ./<path relative to J_halo_dm_lowmass>' (sha256sum -c compatible from this directory).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW_PATTERNS = ("j_mecke_atom.py", "j_mecke_update.py", "j_provenance.py", "j_sums.py", "cluster_poll_mecke.txt",
                "post/mecke_update.json", "post/provenance_checks.json", "post/remote_check_mecke.json",
                "post/upload_mecke.json", "post/remote_check_mecke_checks.json", "post/upload_mecke_checks.json",
                "post/superseded_halfatom_20261016/**/*", "results_mecke/**/*", "results_mecke_checks/**/*",
                "results_mecke_smoke/**/*")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    path = HERE / "SHA256SUMS"
    old = {}
    for line in path.read_text().splitlines():
        if line.strip():
            dig, rel = line.split("  ", 1)
            old[rel[2:] if rel.startswith("./") else rel] = dig
    files = {rel for rel in old if (HERE / rel).is_file()}
    for pat in NEW_PATTERNS:
        for p in HERE.glob(pat):
            rel = p.relative_to(HERE)
            if p.is_file() and "work" not in rel.parts and "__pycache__" not in rel.parts:
                files.add(str(rel))
    new = {rel: sha(HERE / rel) for rel in sorted(files)}
    changed = [{"file": r, "old": old[r], "new": d} for r, d in new.items() if r in old and old[r] != d]
    added = [{"file": r, "new": d} for r, d in new.items() if r not in old]
    removed = [{"file": r, "old": d} for r, d in old.items() if r not in new]
    path.write_text("".join(f"{d}  ./{r}\n" for r, d in new.items()))
    log_p = HERE / "post" / "sums_update_j1.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else {"updates": []}
    log["updates"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n_lines": len(new), "changed": changed,
                           "added": added, "removed": removed})
    log["note"] = ("J1 revision of 2026-10-16 (Mecke quantile maps with the whole DM = 0 atom, maas run results_mecke; "
                   "Q1 cause, Q2 wording and figure fixes): J1's manifest re-hashed over its earlier file set plus the "
                   "revision's files.  Earlier updates of the two shared lines (run_test_j.py, CLUSTER_RUN.json) by J2: "
                   "post_j2/sums_update_j2.json.")
    log_p.write_text(json.dumps(log, indent=1) + "\n")
    print(json.dumps({"lines": len(new), "changed": len(changed), "added": len(added), "removed": len(removed)}))


if __name__ == "__main__":
    main()
