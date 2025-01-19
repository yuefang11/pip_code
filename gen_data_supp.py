# from functions_1214 import *
from functions_for_suppdata_1217 import *
import multiprocessing as mp
import os
import time
import pickle

num_int_list = [300, 1000, 1500]
N_list = [3000, 10000, 15000]
num_each_list = [75, 250, 375]

# num_int_list = [5000]
# N_list = [50000]
# num_each_list = [1250]



# num_workers = 4

K = 4
P = 30
path = 'C:/Users/yuefang/Dropbox/policy_learning_inequality/code/data_supp_250105/'

random.seed(42)
np.random.seed(42)

for (num_int, num_each, N) in zip(num_int_list, num_each_list, N_list):
    random.seed(42)
    np.random.seed(42)
    data1 = dgp_1(N, num_each, P, K, error="uniform")
    score1 = calScoreMulti_est(data1)
    with open(f'{path}data1_uniform_{num_int}.pickle', 'wb') as f:
        pickle.dump(data1, f)
    del data1
    with open(f'{path}score1_uniform_{num_int}.pickle', 'wb') as f:
        pickle.dump(score1, f)
    del score1
    random.seed(121)
    np.random.seed(121)
    data1 = dgp_1(N, num_each, P, K, error="normal")
    score1 = calScoreMulti_est(data1)
    with open(f'{path}data1_normal_{num_int}.pickle', 'wb') as f:
        pickle.dump(data1, f)
    del data1
    with open(f'{path}score1_normal_{num_int}.pickle', 'wb') as f:
        pickle.dump(score1, f)
    del score1

    # random.seed(42)
    # np.random.seed(42)
    # data2 = dgp_2(N, num_each, P, K, error="uniform")
    # score2 = calScoreMulti_est(data2)
    # with open(f'{path}data2_uniform_{num_int}.pickle', 'wb') as f:
    #     pickle.dump(data2, f)
    # del data2
    # with open(f'{path}score2_uniform_{num_int}.pickle', 'wb') as f:
    #     pickle.dump(score2, f)
    # del score2

    # random.seed(42)
    # np.random.seed(42)
    # data2 = dgp_2(N, num_each, P, K, error="normal")
    # score2 = calScoreMulti_est(data2)
    # with open(f'{path}data2_normal_{num_int}.pickle', 'wb') as f:
    #     pickle.dump(data2, f)
    # del data2
    # with open(f'{path}score2_normal_{num_int}.pickle', 'wb') as f:
    #     pickle.dump(score2, f)
    # del score2
    

