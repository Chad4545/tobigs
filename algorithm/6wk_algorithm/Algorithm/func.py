import multiprocessing

def worker(data):
    return data.split(',')

def print_initial_msg():
    print("Start Process : %s" % multiprocessing.current_process().name)