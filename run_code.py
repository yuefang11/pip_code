# from main_function_running import * 
# from functions_1214 import *
from functions_for_suppdata import *
import multiprocessing
import os


# num_int_list = [100, 300, 500, 1000]
# N_list = [1000, 3000, 5000, 10000]
# num_each_list = [25, 75, 125, 250]
# alpha_list = [0.7, 0.5]
# frac_list = [0.4, 0.6]
# time_limit_list = [600, 1800, 3600, 3600]

# num_int_list = [100, 500, 1000, 1500]
# N_list = [1000, 5000, 10000, 15000]
# num_each_list = [25, 125, 250, 375]
# frac_list = [0.4, 0.6]
# time_limit_list = [3600, 3600, 3600, 3600]
# # cnt_list = range(5)
# cnt_list = [2]
# alpha_list = [0.7, 0.5]
num_int_list = [300, 500, 1000, 1500]
N_list = [3000, 5000, 10000, 15000]
num_each_list = [75, 125, 250, 375]
frac_list = [0.4]
time_limit_list = [3600, 3600, 3600, 3600]
# cnt_list = range(5)
cnt_list = [1, 2]
alpha_list = [0.5]
# num_int_list = [100]
# N_list = [1000]
# num_each_list = [25]
# alpha_list = [0.7]
# frac_list = [0.6]
# time_limit_list = [600]

# cnt_list = range(1)
# with multiprocessing.Pool() as pool:

#     pool.map(myparallel, range(5))

# num_int_list = [500]
# N_list = [5000]
# num_each_list = [125]
# frac_list = [0.4, 0.6]
# time_limit_list = [3600]
# cnt_list = range(3, 5)
# alpha_list = [0.7, 0.5]


cwd = os.getcwd()
print(cwd)
path = 'C:/Users/yuefang/Dropbox/policy_learning_inequality/code/results_supplement_250107'
os.chdir(path)
for count in cnt_list:
    t0 = time.time()
    print(count)
    run_main(count, num_int_list, N_list, num_each_list, time_limit_list, alpha_list, frac_list)
    t1 = time.time()
    print('time of the', count, 'dataset is:', t1-t0)

