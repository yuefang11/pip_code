from main_functions_unconstrained_240513 import * 

import multiprocessing
import os


# num_int_list = [100, 300, 500, 1000, 1500]
# N_list = [1000, 3000, 5000, 10000, 15000]
# num_each_list = [25, 75, 125, 250, 375]
# # alpha_list = [0.7, 0.5]
# frac_list = [0.4, 0.6, 0.8]
# time_limit_list = [600, 1800, 3600, 3600, 3600]

num_int_list = [100, 300, 500, 1000, 1500, 5000]
N_list = [1000, 3000, 5000, 10000, 15000, 50000]
num_each_list = [25, 75, 125, 250, 375, 1250]
frac_list = [0.4, 0.6, 0.8]
time_limit_list = [3600, 3600, 3600, 3600, 3600, 10800]
cnt_list = range(5)

# num_int_list = [1000, 1500, 5000]
# N_list = [10000, 15000, 50000]
# num_each_list = [250, 375, 1250]
# frac_list = [0.4, 0.6, 0.8]
# time_limit_list = [3600, 3600, 10800]
# cnt_list = range(5)

# num_int_list = [300, 500, 1000, 1500]
# N_list = [3000, 5000, 10000, 15000]
# num_each_list = [75, 125, 250, 375]
# frac_list = [0.4, 0.6, 0.8]
# time_limit_list = [1800, 3600, 3600, 3600]





# cnt_list = range(1)
# with multiprocessing.Pool() as pool:

#     pool.map(myparallel, range(5))
cwd = os.getcwd()
print(cwd)
path = 'C:/Users/yuefang/Dropbox/policy_learning_inequality/code/results_unconstrained_240629'
os.chdir(path)
for count in cnt_list:
    t0 = time.time()
    print(count)
    run_main_un(count, num_int_list, N_list, num_each_list, time_limit_list, frac_list)
    t1 = time.time()
    print('time of the', count, 'dataset is:', t1-t0)

