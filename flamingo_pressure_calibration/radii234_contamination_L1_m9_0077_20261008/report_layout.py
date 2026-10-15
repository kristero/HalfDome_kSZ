"""Create the all-central report from completed, verified compact products."""
import io
import hashlib
import json
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OUTPUT=HERE/"FLAMINGO_2_3_4R200c_contamination_methodology_report.pdf"
W,H=landscape(A4)
FONTS=Path("/usr/share/fonts/truetype/dejavu")
for name,file in [("Text","DejaVuSans.ttf"),("Bold","DejaVuSans-Bold.ttf"),("Code","DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/file)))


def read(path):
    return json.loads(path.read_text())


class Report:
    def __init__(self):
        self.c=canvas.Canvas(str(OUTPUT),pagesize=(W,H))
        self.page=0
        self.c.setTitle("FLAMINGO: radius comparison and full-snapshot methodology")
        self.c.setAuthor("HalfDome research")
    def start(self,title,subtitle=""):
        if self.page:
            self.c.showPage()
        self.page+=1
        self.c.setFillColorRGB(.04,.16,.25)
        title_size=min(20,20*(W-70)/max(pdfmetrics.stringWidth(title,"Bold",20),1))
        self.c.setFont("Bold",title_size)
        self.c.drawString(35,H-42,title)
        self.c.setFont("Text",10)
        self.c.setFillColorRGB(.35,.35,.35)
        self.c.drawString(36,H-62,subtitle)
        self.c.setStrokeColorRGB(.8,.83,.85)
        self.c.line(35,34,W-35,34)
        self.c.setFont("Text",9)
        self.c.drawString(36,20,"HalfDome | L1_m9 snapshot 0077 | 8 October 2026")
        self.c.drawRightString(W-35,20,str(self.page))
        return H-88
    def text(self,text,y,width=116,size=10.5,leading=15):
        self.c.setFont("Text",size)
        self.c.setFillColorRGB(.08,.08,.08)
        for paragraph in text.split("\n"):
            for line in textwrap.wrap(paragraph,width=width,break_on_hyphens=False):
                if y<42:
                    raise RuntimeError("Text crosses page footer")
                self.c.drawString(36,y,line)
                y-=leading
            y-=7
        return y
    def table(self,head,rows,y,widths,size=10,height=26):
        self.c.setFillColorRGB(.9,.94,.97)
        self.c.rect(36,y-height+7,sum(widths),height,fill=1,stroke=0)
        for r,row in enumerate([head]+rows):
            x=36
            for value,width in zip(row,widths):
                self.c.setFont("Bold" if r==0 else "Text",size)
                self.c.setFillColorRGB(.08,.08,.08)
                self.c.drawString(x+4,y-r*height,str(value))
                x+=width
        return y-(len(rows)+1)*height-9
    def equation(self,latex,y):
        fig=plt.figure(figsize=(11,.6))
        fig.text(.01,.35,latex,fontsize=19)
        buffer=io.BytesIO()
        fig.savefig(buffer,format="png",dpi=190,bbox_inches="tight",transparent=True)
        plt.close(fig)
        buffer.seek(0)
        image=ImageReader(buffer)
        iw,ih=image.getSize()
        scale=min((W-80)/iw,50/ih)
        self.c.drawImage(image,40,y-ih*scale,width=iw*scale,height=ih*scale,mask="auto")
        return y-62
    def figure(self,path,y):
        image=ImageReader(str(path))
        iw,ih=image.getSize()
        scale=min((W-72)/iw,(y-87)/ih)
        width,height=iw*scale,ih*scale
        self.c.drawImage(image,(W-width)/2,y-height,width=width,height=height,mask="auto")
    def finish(self):
        self.c.save()

