"""Check documentation provenance and render contact sheets for visual review.

Run after build_walkthrough.py. The --reviewed flag records the operator's
completed visual inspection; it must not substitute for actually viewing pages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader

from walkthrough_examples import DEFAULT_STUDY


HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parents[1] / "tmp/pdfs/procedure_walkthrough_20261008"
PDF = HERE / "FLAMINGO_snapshot_to_stack_fits_walkthrough.pdf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", action="store_true")
    args = parser.parse_args()
    evidence_path = HERE / "walkthrough_verification.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["report_sha256"] == digest(PDF)
    pages = PdfReader(PDF).pages
    assert len(pages) == 12
    for page in pages:
        text = page.extract_text()
        assert len(text) > 500
        assert "\ufffd" not in text and "\x00" not in text

    # Check source inputs against the completed study's frozen delivery manifest.
    frozen = json.loads((DEFAULT_STUDY / "delivery_manifest.json").read_text())
    for name, expected in evidence["source_sha256"].items():
        assert digest(DEFAULT_STUDY / name) == expected, name
        assert frozen["files"][name]["sha256"] == expected, name
    evidence["used_sources_match_original_delivery_manifest"] = True
    evidence["used_sources_checked"] = len(evidence["source_sha256"])

    SCRATCH.mkdir(parents=True, exist_ok=True)
    # Always render the current PDF; existing PNGs can belong to an earlier draft.
    subprocess.run(["pdftoppm", "-r", "120", "-png", str(PDF), str(SCRATCH / "page")], check=True)
    rendered = sorted(SCRATCH.glob("page-*.png"))
    assert len(rendered) == 12
    for start in range(0, 12, 4):
        sheet = Image.new("RGB", (1800, 1320), "#d8e0e7")
        draw = ImageDraw.Draw(sheet)
        for j, filename in enumerate(rendered[start : start + 4]):
            page = Image.open(filename).convert("RGB")
            page.thumbnail((880, 625))
            x, y = (j % 2) * 900 + 10, (j // 2) * 660 + 25
            sheet.paste(page, (x, y))
            draw.text((x, y - 17), f"Page {start + j + 1}", fill="black")
        sheet.save(SCRATCH / f"contact_{start//4+1}.png")

    evidence["rendered_pages"] = 12
    evidence["visual_review_complete"] = args.reviewed
    evidence["new_source_sha256"] = {
        name: digest(HERE / name)
        for name in ("build_walkthrough.py", "walkthrough_examples.py", "verify_walkthrough.py", "README.txt")
    }
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"pages": 12, "source_hash_checks": evidence["used_sources_checked"],
                      "visual_review_complete": args.reviewed, "scratch": str(SCRATCH)}))


if __name__ == "__main__":
    main()
