import time

def measure_runtime(function, *args, **kwargs):
    start = time.time()
    result = function(*args, **kwargs)
    end = time.time()
    return result, end - start
