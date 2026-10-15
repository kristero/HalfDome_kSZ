"""Render every report page and check text bounds before manual visual review."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw
from pypdf import PdfReader
from report_layout import OUTPUT

HERE=Path(__file__).resolve().parent
SCRATCH=HERE.parents[1]/'tmp/pdfs/radii234_contamination_20261008'


def main():
    SCRATCH.mkdir(parents=True,exist_ok=True)
    reader=PdfReader(OUTPUT);pages=len(reader.pages)
    bbox=SCRATCH/'text_bounds.xhtml'
    subprocess.run(['pdftotext','-bbox-layout',str(OUTPUT),str(bbox)],check=True)
    tree=ET.parse(bbox);bad=[]
    for i,page in enumerate(tree.findall('.//{*}page')):
        width=float(page.attrib['width']);height=float(page.attrib['height'])
        for word in page.findall('.//{*}word'):
            a=word.attrib
            if float(a['xMin'])<30 or float(a['xMax'])>width-29 or float(a['yMin'])<20 or float(a['yMax'])>height-12:
                bad.append({'page':i+1,'word':word.text,'bounds':a})
    if bad:
        (HERE/'pdf_bounds_failures.json').write_text(json.dumps(bad,indent=2)+'\n')
        raise RuntimeError(f'{len(bad)} text words exceed report margins')
    subprocess.run(['pdftoppm','-r','120','-png',str(OUTPUT),str(SCRATCH/'page')],check=True)
    images=sorted(SCRATCH.glob('page-*.png'));assert len(images)==pages
    contacts=[]
    for start in range(0,pages,4):
        sheet=Image.new('RGB',(1800,1324),'#c9ced2');draw=ImageDraw.Draw(sheet)
        for k,path in enumerate(images[start:start+4]):
            image=Image.open(path).convert('RGB');image.thumbnail((900,637))
            xx=(k%2)*900;yy=(k//2)*662
            sheet.paste(image,(xx,yy+25));draw.text((xx+12,yy+6),f'Page {start+k+1}',fill='black')
        target=SCRATCH/f'contact_{start//4+1:02d}.png';sheet.save(target);contacts.append(str(target))
    result={'pages':pages,'all_pages_rendered':True,'text_margin_checks_passed':True,
        'PDF_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'contacts':contacts,
        'visual_review_complete':False}
    (HERE/'report_visual_qa.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
