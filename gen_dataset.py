from main_function_running import * 
from main_functions_unconstrained_240513 import *
import multiprocessing as mp
import os

# num_int_list = [100, 300, 500, 1000, 1500]
# N_list = [1000, 3000, 5000, 10000, 15000]
# num_each_list = [25, 75, 125, 250, 375]

num_int_list = [5000]
N_list = [50000]
num_each_list = [1250]

cnt_list = range(5)
num_workers = 5

K = 4
P = 20
dim = 30
path = 'C:/Users/yuefang/Dropbox/policy_learning_inequality/code/data_saved/'
os.chdir(path)
def parallel_gen_data(cnt):
    for (num_int, num_each, N) in zip(num_int_list, num_each_list, N_list):
        num_val_list = np.random.uniform(-1, 1, (num_each, P))
        data, score = generate_data_alt(N, dim, K, P, num_val_list)
        with open(f'data_{num_int}_{cnt}.pickle', 'wb') as f:
            pickle.dump(data, f)
        del data
        with open(f'score_{num_int}_{cnt}.pickle', 'wb') as f:
            pickle.dump(score, f)
        del score
def my_worker(cnt):
    print("Dataset %s\tWaiting" % cnt)
    parallel_gen_data(cnt)
    time.sleep(1)
    print("Dataset %s\tDone" % cnt)
def mp_handler():
    p = mp.Pool(num_workers)
    p.map(my_worker, cnt_list)
if __name__ == '__main__':
    mp_handler()
# for (num_int, num_each, N, time_limit) in zip(num_int_list, num_each_list, N_list, time_limit_list):
#     table_count_num_int = collections.defaultdict(dict)
#     num_val_list = np.random.uniform(-1, 1, (num_each, P))

#     for count in cnt_list:
#         data, score = generate_data(N, dim, K, P, num_val_list)
#         with open(f'data_{N}_{count}.pickle', 'wb') as f:
#             pickle.dump(data, f)
#         with open(f'score_{N}_{count}.pickle', 'wb') as f:
#             pickle.dump(score, f) 
