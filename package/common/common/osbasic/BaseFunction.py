from functools import wraps
from datetime import datetime

#時間裝飾器，計算函式執行
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print(func.__name__, end-start)
        return result
    return wrapper

