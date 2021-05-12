from functools import wraps
from datetime import datetime
def timer(func):
    @wraps(func)
    def layer(*args,**kwargs):
        #print(args)
        def count(*args,**kwargs):
            start=datetime.now()
            res=func(*args,**kwargs)
            time=(datetime.now()-start)
            return res,time
        return count(*args,**kwargs)
    return layer