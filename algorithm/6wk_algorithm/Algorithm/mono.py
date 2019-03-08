import numpy as np
import time
import multiprocessing
import tqdm
import os

def fibo(n):
    if n<=1:
        return n
    else:
        return fibo(n-1) + fibo(n-2)

start_time = time.time()
num_list = np.array([31, 32, 33, 34])
result_list = []
for num in num_list:
    result_list.append(fibo(num))

print(result_list)
print("--- %s seconds" % (time.time()-start_time))