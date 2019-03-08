import numpy as np
import time
import multiprocessing
import tqdm
import os

def print_initial_msg():
    print("Start Process : %s" % multiprocessing.current_process().name)

def fibo(n):
    if n<=1:
        return n
    else:
        return fibo(n-1) + fibo(n-2)

def print_fibo(n):
    print(fibo(n))

def main_(data_list):
    start_time = time.time()
    pool = multiprocessing.Pool(processes=4, initializer=print_initial_msg)
    result = pool.map(print_fibo, data_list)
    pool.close()
    pool.join()
    print("Time : %f" % (time.time()-start_time))
    return result


if __name__ == '__main__':
    num_list = np.array([31, 32, 33, 34])
    main_(num_list)