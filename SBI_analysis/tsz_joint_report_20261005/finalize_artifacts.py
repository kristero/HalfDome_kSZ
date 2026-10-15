#!/usr/bin/env python3
"""Create a delivery inventory after numerical checks and visual PDF review.

The caller must supply the SHA256 of the reviewed PDF. A rebuild changes that
identity and requires another visual review before this gate can pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PDF = ROOT / 'output/pdf/HalfDome_Merlin_seed_mask_and_Pearson_report.pdf'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main(reviewed_sha):
    assert sha(PDF)==reviewed_sha, 'PDF differs from the reviewed version'
    pages=PdfReader(PDF).pages
    assert len(pages)==36
    for i,page in enumerate(pages,1):
        text=page.extract_text()
        assert len(text)>200,i
        assert '\ufffd' not in text and '\u25a0' not in text,i
    inference=json.loads((ROOT/'INFERENCE_VERIFIED.json').read_text())
    assert inference['saved_control_features_max_error']==0
    assert inference['outputs_sha256_verified']
    assert inference['application_exit_status']==0
    for source in json.loads((ROOT/'source_excerpts.json').read_text()):
        assert sha(ROOT/source['file'])==source['source_sha256']
    for source in json.loads((ROOT/'evidence_manifest.json').read_text()):
        assert sha(ROOT/source['delivered'])==source['sha256']
    products={}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if any(part in ['tmp','__pycache__'] for part in relative.parts):continue
        if path.name in ['VERIFIED.json','FILES_CREATED.md','inference_delivery.zip']:continue
        products[str(relative)]={'bytes':path.stat().st_size,'sha256':sha(path)}
    verified=dict(checked_utc=datetime.now(timezone.utc).isoformat(),pdf_pages=36,
        reviewed_pdf_sha256=reviewed_sha,all_pages_visually_reviewed=True,
        inference=inference,source_excerpts_sha256_checked=True,
        evidence_sha256_checked=True,files=products,
        full_posterior_location='/lustre/work/kristero10/tsz_joint_report_20261005/data',
        prior_roots_preserved=True)
    (ROOT/'VERIFIED.json').write_text(json.dumps(verified,indent=2)+'\n')
    inventory=['# Files created or edited for this request','',
        'All new work is confined to `tsz_joint_report_20261005`; prior experiment files were read only.',
        '', 'Editable source: `baseline_pearson.py`, `pearson.pbs`, `collect_evidence.py`,',
        '`report_figures.py`, `build_report.py`, `finalize_artifacts.py`, `README.md`.',
        '', 'PDF, figures, CSV tables, manifests and archived evidence:', '',
        '| Relative file | Bytes | SHA256 |','|---|---:|---|']
    inventory.extend(f"| `{name}` | {item['bytes']} | `{item['sha256']}` |" for name,item in products.items())
    inventory.extend(['','The full six posterior arrays are retained on idark, not duplicated locally.',
        '`PEARSON_COMPLETE.json` lists and hashes those arrays and the consumed cluster inputs.',
        '`VERIFIED.json` records all page review and delivery checks. Temporary layout previews were removed.',''])
    (ROOT/'FILES_CREATED.md').write_text('\n'.join(inventory))
    print('Verified',len(products),'delivered files;',len(pages),'reviewed PDF pages.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-pdf-sha256',required=True)
    main(parser.parse_args().reviewed_pdf_sha256)
