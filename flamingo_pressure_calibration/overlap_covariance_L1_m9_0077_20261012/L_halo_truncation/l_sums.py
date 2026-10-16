"""Write SHA256SUMS of the L1 deliverables (code, inputs, results, figures, verification) in L_halo_truncation/."""
from __future__ import annotations

import l_common as LC


def main():
    files = sorted(p for p in LC.HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and "work" not in p.parts
                   and p.name != "SHA256SUMS")
    lines = [f"{LC.sha256(p)}  {p.relative_to(LC.HERE)}" for p in files]
    (LC.HERE / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    LC.log(f"SHA256SUMS: {len(lines)} files")


if __name__ == "__main__":
    main()
