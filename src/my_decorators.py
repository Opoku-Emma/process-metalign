import time
from datetime import datetime as dt

from typing import Callable


def my_timer(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        t0 = dt.now()
        res = func(*args, **kwargs)
        t1 = dt.now()
        run_time = t1 - t0
        print(f"""Start time: {t0}\nEnd time: {t1}\nTotal runtime: {run_time}""")
        return res
    return wrapper

