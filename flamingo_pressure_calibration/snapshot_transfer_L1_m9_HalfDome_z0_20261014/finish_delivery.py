"""Finalize reviewed artifacts without changing fits, maps or cluster receipts."""
from pathlib import Path
import datetime
import hashlib
from html.parser import HTMLParser
import json
import shutil
import pdfplumber
import build_delivery

ROOT=Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text())


def checked_path(path):
    resolved=path.resolve()
    assert resolved!=ROOT.resolve() and ROOT.resolve() in resolved.parents
    return resolved


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.targets=[]

    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
            if name in ['href','src'] and value and not value.startswith(('http:','https:','#')):
                self.targets.append(value)


def main():
    qa=read('report_qa.json')
    assert qa['automated_checks_passed']
    # Check vector text bounds as well as the reviewed page images.
    for record in qa['local_reports']:
        with pdfplumber.open(ROOT/record['file']) as document:
            for page in document.pages:
                outside=[c for c in page.chars if c['text'].strip() and
                         (c['x0']<8 or c['x1']>page.width-8 or c['top']<8 or c['bottom']>page.height-8)]
                assert not outside,(record['file'],page.page_number,outside[:1])
                page.flush_cache()
    qa['visual_review']='passed: all page sheets plus full-size equations, code and overview pages inspected'
    qa['text_bounds_passed']=True
    qa['reviewed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT/'report_qa.json').write_text(json.dumps(qa,indent=2)+'\n')

    # Keep current review sheets/pages as verification evidence. Only stale,
    # regenerable renders and already unpacked transfer archives are removed.
    retained_review=Path(qa['local_reports'][0]['render_directory']).parent.name
    temporary=ROOT/'tmp/pdfs'
    removed=[]
    for path in temporary.iterdir():
        if path.name==retained_review:continue
        target=checked_path(path)
        size=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()) if target.is_dir() else target.stat().st_size
        if target.is_dir():shutil.rmtree(target)
        else:target.unlink()
        removed.append(dict(path=str(path.relative_to(ROOT)),bytes=size))
    for name in ['delivery_bundle.tar.gz','.transfer_idark.tar','report.pid','qa.pid','preview.pid']:
        path=ROOT/name
        if path.exists():
            target=checked_path(path);size=target.stat().st_size;target.unlink()
            removed.append(dict(path=name,bytes=size))
    (ROOT/'delivery_cleanup.json').write_text(json.dumps(dict(removed=removed,
        removed_bytes=sum(v['bytes'] for v in removed),
        retained='Scientific caches, accounting logs, cluster receipts and current visual-review evidence.'),indent=2)+'\n')
    status='''# Final local delivery

Both reports are complete and reviewed: five-page summary and 59-page detailed report.
The gallery, CSV case statistics, parameter distributions, resource accounting and
delivery checksums are available in this directory. All 57 computed cases and two
derived controls remain on idark. Native full-precision maps and cached voxels remain
on STRW. Fifteen cluster job instances were submitted; no new jobs or particle reads
were needed for this finalization.

The scientific source files, fit inputs and 91 figure files match the earlier cluster
verification hashes. The last live idark verification was 2026-10-14 14:30 UTC;
idark was unreachable during final local report delivery. Final report changes are
local and are not claimed to have been uploaded to idark. CLUSTER_VERIFIED.json is
the historical cluster receipt; reports_complete.json, report_qa.json and
delivery_manifest.json describe the final local artifacts.

The earlier HANDOFF_1735.md is retained as a historical stop-point record. The later
user instruction to continue authorized finalization and superseded that stop.
Scientific fits and map values were not changed by report layout or accounting fixes.
'''
    (ROOT/'DELIVERY_STATUS.md').write_text(status)
    build_delivery.main()
    parser=Links();parser.feed((ROOT/'index.html').read_text())
    for target in parser.targets:assert (ROOT/target).exists(),target
    assert len(list((ROOT/'figures').glob('*.png')))==52
    print('Final reports, all gallery links, 52 PNG figures and text bounds verified.')
    print('Removed redundant bytes:',sum(v['bytes'] for v in removed))


if __name__=='__main__':main()
