import inspect
from classy_sz import Class
for name in dir(Class):
    if any(part in name for part in ['pressure','battaglia','sz_at','r200']):
        obj=getattr(Class,name)
        try: signature=str(inspect.signature(obj))
        except Exception: signature=''
        print(name,signature,str(getattr(obj,'__doc__',''))[:1400])
