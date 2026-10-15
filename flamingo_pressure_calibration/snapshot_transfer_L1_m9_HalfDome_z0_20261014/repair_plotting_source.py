"""Recover the interrupted plotting edit from a verified nonempty copy."""
from pathlib import Path
import hashlib
import py_compile

root=Path(__file__).resolve().parent
source=root/'aggregate_recovered.py'
text=source.read_text(encoding='utf-8-sig')
assert len(text)>20000 and 'def main()' in text, 'Recovery copy is incomplete.'
text=text.replace("layout='constrained'",'constrained_layout=True')
text=text.replace("'602976.idark',os.environ", "'602976.idark','603000.idark','603002.idark',os.environ")
assert "layout='constrained'" not in text
target=root/'aggregate_and_plot.py'
temporary=root/'aggregate_repaired.tmp'
temporary.write_text(text)
py_compile.compile(str(temporary),doraise=True)
temporary.replace(target)
print('recovered',target.stat().st_size,hashlib.sha256(target.read_bytes()).hexdigest())
