"""Check cached scientific hashes, render every PDF page and build QA sheets."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import signal
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
renderer=None


def sha256(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def stop(signum,frame):
    if renderer is not None and renderer.poll() is None:
        renderer.terminate()
    raise SystemExit('Report QA stopped by memory guard.')


def main():
    global renderer
    (ROOT/'qa.pid').write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM,stop)
    cluster=json.loads((ROOT/'CLUSTER_VERIFIED.json').read_text())
    science=['physics.py','prepare_idark.py','painting.py','run_comparisons.py',
             'field_statistics.py','verify_numerics.py']
    for name in science:
        assert sha256(ROOT/name)==cluster['code_hashes'][name],name
    for name in ['fits.json','fits_sem_weighting_control.json','mass_mapping.json',
                 'measured_flamingo_compact.h5']:
        assert sha256(ROOT/name)==cluster['frozen_input_hashes'][name],name
    figures=json.loads((ROOT/'cluster_analysis_complete.json').read_text())['figures']
    for name,value in figures.items():assert sha256(ROOT/name)==value,name
    reports=json.loads((ROOT/'reports_complete.json').read_text())['outputs']
    root_review=ROOT/'tmp/pdfs'/('review_'+reports[-1]['sha256'][:12])
    root_review.mkdir(parents=True,exist_ok=True)
    records=[]
    for record in reports:
        pdf=ROOT/record['file'];assert sha256(pdf)==record['sha256']
        pages=PdfReader(pdf).pages
        assert len(pages)==record['pages']
        text=[page.extract_text() or '' for page in pages]
        assert all(len(t)>100 for t in text),'An empty or unusually sparse page needs review.'
        assert not any('\ufffd' in t for t in text),'Unsupported text glyph.'
        name='summary' if 'summary' in pdf.name else 'detailed'
        target=root_review/name;target.mkdir(exist_ok=True)
        renderer=subprocess.Popen(['pdftoppm','-r','90','-png',str(pdf),str(target/'page')])
        assert renderer.wait()==0
        renderer=None
        images=sorted(target.glob('page-*.png'))
        assert len(images)==len(pages)
        # Twelve pages per sheet keeps paragraphs and equations inspectable.
        sheets=[]
        for start in range(0,len(images),12):
            block=images[start:start+12]
            columns=3;cell_w,cell_h=560,420
            sheet=Image.new('RGB',(columns*cell_w,((len(block)+columns-1)//columns)*cell_h),'#dce3e9')
            draw=ImageDraw.Draw(sheet)
            for offset,path in enumerate(block):
                with Image.open(path) as page:
                    page.thumbnail((cell_w-10,cell_h-25))
                    x=(offset%columns)*cell_w+5;y=(offset//columns)*cell_h+22
                    sheet.paste(page,(x,y));draw.text((x,y-18),str(start+offset+1),fill='black')
            path=root_review/('%s_sheet_%02d.jpg'%(name,start//12+1))
            sheet.save(path,quality=92)
            sheets.append(str(path.relative_to(ROOT)))
        records.append(dict(file=record['file'],pages=len(pages),sha256=record['sha256'],
                            render_directory=str(target.relative_to(ROOT)),contact_sheets=sheets))
    result=dict(completed=True,automated_checks_passed=True,visual_review='pending',
                scientific_sources_unchanged=science,verified_figure_files=len(figures),
                last_live_cluster_verification_utc=cluster['timestamp_utc'],
                current_idark_access=False,local_reports=records)
    (ROOT/'report_qa.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
