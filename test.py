import time
from numba import jit, njit
from functools import wraps




N= int(1e6)

def app():
    l = []
    for i in range(N):
        if i % 2:
            l.append(i)
    return l



def app_jit():
    l = []
    for i in range(N):
        if i % 2:
            l.append(i)
    return l

@njit
def app_njit():
    l = []
    for i in range(N):
        if i % 2:
            l.append(i)
    return l


t = time.perf_counter()
app()
print(round(time.perf_counter() - t, 2))

t = time.perf_counter()
app_jit()
print(round(time.perf_counter() - t, 2))

t = time.perf_counter()
app_njit()
print(round(time.perf_counter() - t, 2))
