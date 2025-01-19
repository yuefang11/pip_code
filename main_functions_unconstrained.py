import numpy as np
import pandas as pd
import math
import time
import gurobipy as gp
import random
import matplotlib.pyplot as plt
import collections
import pickle
import os
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from main_function_running import *

def calScoreMulti_est_un(data):
    Y, df_D, A, prob, df_cov, df_X, df_Z, treat_max = data
    P = df_X.shape[1]
    dim = df_cov.shape[1]
    N = len(Y)
    K = len(np.unique(A))
    M = np.max(Y) - np.min(Y)

    df_DY = pd.DataFrame(df_D*np.array(Y)[:,np.newaxis])
    df_DY = df_DY.add_prefix('DY_')
    df_all = pd.concat([df_cov, df_D, df_DY], axis = 1)
    df_all["Y"] = Y
    df_all["num"] = np.ones(N)
    group_df_D = df_all.groupby(list(df_X.columns)).sum()
    num_val_dim = group_df_D.shape[0] # count the number of unique covariate combinations
    prob_est = np.array(group_df_D[df_D.columns.tolist()])/np.array(group_df_D["num"])[:, np.newaxis]
    print(np.sum(prob_est==0))
    if np.sum(prob_est==0) > 0:
        return None
    prob_est_df = (group_df_D[df_D.columns.tolist()]/np.array(group_df_D["num"])[:, np.newaxis]).reset_index()
    prob_est_expand = prob_est[:, np.newaxis, :, np.newaxis]
    prob_est_expand = prob_est_expand * prob_est_expand.transpose((3, 0, 1, 2))

    x =  np.array(prob_est_df.iloc[:,:P]).reshape(num_val_dim, P)    
    score_unit = np.array(group_df_D[df_DY.columns.tolist()]/prob_est)
    score_pair = None
    return (x, score_pair, score_unit, M, N)

def optimization_global_un(b0_init, b0, score,  lamd, time_limit, count):
    t0 = time.time()

    x, score_pair, score_unit, M, N = score
    B = 1
    #B_bar = np.min(-B*np.sum(abs(x), axis=1))
    B_bar = -2 * B * np.sum(abs(x), axis=1)[:, np.newaxis] + b0_init - 1 - 0.001
    x_card = score_unit.shape[0]
    K = score_unit.shape[1]
    P = x.shape[1]
    num_int_global = K * x_card
    # build the gurobi model, set the parameters
    model = gp.Model()
    # model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 0.01)
    model.params.LogFile=f'model_{count}_{num_int_global}.log'
    # add the variables
    beta = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, lb=-1, ub = 1, name="beta")
    z = model.addVars(x_card, K, vtype=gp.GRB.BINARY, name="z")

    
    xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta")
    # max_xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta_max")
    t = model.addVar(vtype=gp.GRB.CONTINUOUS, name="t")
    s = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, name="s")
    if b0 is not None:
        for k in range(K):
            for p in range(P):
                beta[k,p].Start = b0[k][p]

    # set the objective
    model.setObjective(gp.quicksum(z[x_idx, j_idx] * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))/N - lamd * t, sense = gp.GRB.MAXIMIZE)
    model.addConstrs((beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstrs((-beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstr(gp.quicksum(s[j, p] for j in range(K) for p in range(P)) == t )
    # set the constraints
    for x_idx in range(x_card):
        for j_idx in range(K):
            model.addConstr(xbeta[x_idx, j_idx] - (gp.quicksum(beta[j_idx, p] * x[x_idx][p] for p in range(P)) + b0_init[j_idx]) == 0)          
 
    
    for x_idx in range(x_card):
        for j_idx in range(K):
            for k in range(K):
                if k != j_idx:
                    model.addConstr(xbeta[x_idx, j_idx] - xbeta[x_idx, k] - 0.001  >=  B_bar[x_idx, j_idx] * (1 - z[x_idx, j_idx]))

    model.update()
    model.optimize()

    # check the objective

    obj = model.objVal
    print(obj)


    supp = []
    for x_idx in range(x_card):
        supp.append(gp.quicksum(z[x_idx, j_idx].x for j_idx in range(K)).getValue())
    supp = np.array(supp)
    print(np.sum(supp==0), np.sum(np.abs(supp-1)<0.01), np.sum(supp-1>=0.01))
    
    results = []
    for v in model.getVars():
        results.append(v.x)
    b1 = np.array(results[:K*(P)])
    b1 = b1.reshape(K, P)
    
    xbeta_res = np.dot(x, b1.T) + b0_init
    f = np.zeros((x_card, K))
    for i in range(xbeta_res.shape[1]):
        f[:,i] = xbeta_res[:,i] - np.max(np.delete(xbeta_res, i, axis=1), axis = 1) - 0.001 
    f_score = (f >= 0) * score_unit    
    f_score_avg = np.mean(f_score, axis = 1)
    
    b_norm = np.sqrt(np.sum(b1**2))
    non_distinguish = 1 * (f/b_norm < 0.001) * (f/b_norm > -0.001)
    print(np.sum(non_distinguish))
    
    
    t1 = time.time()
    time_used = t1 - t0 
    print(time_used)
    return (b1, obj, time_used)
        
def optimization_linear_un(b0_init, score,  lamd):
    t0 = time.time()
    x, score_pair, score_unit, M, N = score
    B = 1
    #B_bar = np.min(-B*np.sum(abs(x), axis=1))
    B_bar = -2 * B * np.sum(abs(x), axis=1)[:, np.newaxis] + b0_init - 1 - 0.001
    x_card = score_unit.shape[0]

    K = score_unit.shape[1]
    P = x.shape[1]

    model = gp.Model()
    model.setParam('OutputFlag', 0)
    # model.setParam('NonConvex', 2)
    model.setParam('TimeLimit', 600)

    beta = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, lb=-1, ub = 1, name="beta")
    z = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, lb = 0, ub = 1, name="z")
    
    
    xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta")
    # max_xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta_max")
    t = model.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, name="t")
    s = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, name="s")
    

    model.setObjective(gp.quicksum(z[x_idx, j_idx] * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))/N- lamd * t, sense = gp.GRB.MAXIMIZE)
    model.addConstrs((beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstrs((-beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstr(gp.quicksum(s[j, p] for j in range(K) for p in range(P)) == t )
    for x_idx in range(x_card):
        for j_idx in range(K):
            model.addConstr(xbeta[x_idx, j_idx] - (gp.quicksum(beta[j_idx, p]*x[x_idx][p] for p in range(P)) + b0_init[j_idx]) == 0)
            
    for x_idx in range(x_card):
        for j_idx in range(K):
            for k in range(K):
                if k != j_idx:
                    model.addConstr(xbeta[x_idx, j_idx] - xbeta[x_idx, k]- 0.001   >=  B_bar[x_idx, j_idx] * (1 - z[x_idx, j_idx]))
            
    model.update()
    model.optimize()
    
    results = []
    for v in model.getVars():
        results.append(v.x)
    b1 = np.array(results[:K*(P)])
    b1 = b1.reshape(K, P)
        
    xbeta_res = np.dot(x, b1.T) + b0_init
    f = np.zeros((x_card, K))
    for i in range(xbeta_res.shape[1]):
        f[:,i] = xbeta_res[:,i] - np.max(np.delete(xbeta_res, i, axis=1), axis = 1) - 0.001 
    f_score = (f >= 0) * score_unit    
    f_score_avg = np.mean(f_score, axis = 1)
    
    b_norm = np.sqrt(np.sum(b1**2))
    non_distinguish = 1 * (f/b_norm < 0.001) * (f/b_norm > -0.001)
    print(np.sum(non_distinguish))
    
    # print(f_score)
    # print(f_score_avg)
    
    obj = model.objVal
    treat = np.where(f >= 0)
    
    print(obj)
    print(t.x)    
    t1 = time.time()
    time_used = t1 - t0
    print(time_used)
    return (b1, obj, time_used)



def optimization_un(t0_init,  max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2,  lamd, tol, cnt = 0, obj = 0, iter = 0):
    x, score_pair, score_unit, M, N = score
    print(cnt, iter, epsilon1, epsilon2)
    
    B = 1
    # B_bar = np.min(-B*np.sum(abs(x), axis=1))
    
    x_card = score_unit.shape[0]
    K = score_unit.shape[1]
    P = x.shape[1]
    xbeta = np.dot(x, b0.T) + b0_init
    f = np.zeros((x_card, K))
    for i in range(xbeta.shape[1]):
        f[:,i] = xbeta[:,i] - np.max(np.delete(xbeta, i, axis=1), axis = 1) - 0.001 
    
    b_norm = np.sqrt(np.sum(b0**2))
    in_between = 1 * (f < epsilon1) * (f > -epsilon2)
    num_integer = np.sum(in_between)
    print(num_integer)
    
    if num_integer > x_card * K * frac:
        epsilon1 = epsilon1 * 4/5
        epsilon2 = epsilon2 * 4/5
        return optimization_un(t0_init,  max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2,  lamd, tol, 0, obj, iter)
    
    ans["cnt"].append([cnt, iter])
    ans["epsilon"].append([-epsilon2, epsilon1])
    ans["num_integer"].append(num_integer)
    non_distinguish = 1 * (f/b_norm < 0.001) * (f/b_norm > -0.001)
    print(np.sum(non_distinguish))
    ans["non_distinguish"].append(np.sum(non_distinguish))
    z_idx = np.argwhere(in_between == 1)
    
    right = 1 * (f > epsilon1)
    left = 1 * (f < -epsilon2)
    right_idx = np.argwhere(right == 1)
    left_idx = np.argwhere(left == 1)
    if cnt == 0 and iter == 0:
        obj_init = np.sum((f >= 0) * score_unit)/N
    else: 
        obj_init = obj
    
    other_idx = np.argwhere(in_between == 0)
      
    B_bar_all = -2 * B * np.sum(abs(x), axis=1)[:, np.newaxis] + b0_init - 1 - 0.001
    #B_bar = B_bar_all[z_idx[:,0], z_idx[:,1]]

    right_idx_dic = collections.defaultdict(list)
    z_idx_dic = collections.defaultdict(list)
    for i in range(len(right_idx)):
        right_idx_dic[right_idx[i][0]].append(right_idx[i][1])
    for i in range(len(z_idx)):
        z_idx_dic[z_idx[i][0]].append(z_idx[i][1])


    # Const_pair = np.sum([score_pair[right_idx[i][0]][right_idx[j][0]][right_idx[i][1]][right_idx[j][1]] for i in range(len(right_idx)) for j in range(len(right_idx))]) 
    Const_unit = np.sum([score_unit[right_idx[i][0]][right_idx[i][1]] for i in range(len(right_idx))])
    # print("Const unit number is", Const_unit)
    t0 = time.time()
    # build the model and set the parameters
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 120)
    # model.setParam('MIPGap', 0.01)
    if obj_init >= 0:
        model.params.BestObjStop = obj_init + 0.1
    
    # add the variables
    beta = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, lb=-1, ub = 1, name="beta")
    z = model.addVars(np.arange(len(z_idx)), vtype=gp.GRB.BINARY, name="z")
    # eta = model.addVars(np.arange(len(z_idx)), np.arange(len(z_idx)), vtype=gp.GRB.CONTINUOUS, lb = 0, name="eta")
    # gamma_var = model.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, name="gamma")
    xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta")
    # max_xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta_max")
    t = model.addVar(vtype=gp.GRB.CONTINUOUS, name="t")
    s = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, name="s")
    for k in range(K):
        for p in range(P):
            beta[k,p].Start = b0[k][p]
    
    # set the objective
    model.setObjective((gp.quicksum(z[i] * score_unit[z_idx[i][0], z_idx[i][1]] for i in range(len(z_idx))) +  Const_unit)/N  - lamd * t, sense = gp.GRB.MAXIMIZE)
    model.addConstrs((beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstrs((-beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstr(gp.quicksum(s[j, p] for j in range(K) for p in range(P)) == t )
    # set the constraints
    for x_idx in range(x_card):
        for j_idx in range(K):
            model.addConstr(xbeta[x_idx, j_idx] - (gp.quicksum(beta[j_idx, p]*x[x_idx][p] for p in range(P)) + b0_init[j_idx]) == 0)
    
    
    for i in range(len(z_idx)):
        x_idx = z_idx[i][0]
        j_idx = z_idx[i][1]
        for k in range(K):
            if k != j_idx:
                model.addConstr(xbeta[x_idx, j_idx] - xbeta[x_idx, k] - 0.001 >=  B_bar_all[x_idx, j_idx] * (1 - z[i]))
        # model.addConstr(xbeta[x_idx, j_idx] - max_xbeta[x_idx, j_idx] - 0.001 >=  B_bar *(1 - z[i]))

    for i in range(len(right_idx)):
        x_idx = right_idx[i][0]
        j_idx = right_idx[i][1]
        for k in range(K):
            if k != j_idx:
                model.addConstr(xbeta[x_idx, j_idx] - xbeta[x_idx, k]  - 0.001 >=  0)
        # model.addConstr(xbeta[x_idx, j_idx] - max_xbeta[x_idx, j_idx] - 0.001  >= 0)
        
    model.update()
    model.optimize()
    obj = model.objVal
    print(obj_init, obj)
    print(t.x)
    t1 = time.time()
    time_used = t1 - t0
    results = []
    for v in model.getVars():
        results.append(v.x)
    
    z_val = []
    for i in range(len(z_idx)):
        z_val.append(z[i].x)
    
    b1 = np.array(results[:K*(P)])
    b1 = b1.reshape(K, P)
    # if gamma_var.x == 0:
    

    
    ans["obj_init"].append(obj_init)
    ans["obj"].append(obj)
    ans["welfare"].append(obj)
    ans["time"].append(t1-t0)
    

    print(time_used)
    cur_time = time.time()
    if iter >= max_iter or cur_time - t0_init > 3600:
        return (b1, obj, time_used, z_idx)
    if abs(obj - obj_init) <= tol or cnt >= 10:
        b0 = b1
        epsilon1 = epsilon1 * 2
        epsilon2 = epsilon2 * 2
        iter += 1
        return optimization_un(t0_init,  max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2,  lamd, tol, 0, obj, iter) 
    else:
        b0 = b1
        epsilon1 = epsilon1 * (3/4)
        epsilon2 = epsilon2 * (3/4)
        return optimization_un(t0_init,  max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2,  lamd, tol, cnt+1, obj, iter)



def evaluate_un(b1, covariates_X, treat_max):
    xbeta_res = np.dot(covariates_X, b1.T)
    f_score = np.zeros(xbeta_res.shape)
    for i in range(xbeta_res.shape[1]):
        f_score[:,i] = xbeta_res[:,i] - np.max(np.delete(xbeta_res, i, axis=1), axis = 1) 
    treat_opt = np.argmax(f_score, axis = 1)
    correct_assign = np.mean(treat_opt == treat_max)
    return correct_assign

def main_un(b0_init, b0, score, covariates_X, treat_max, initial_time,  lamd, max_iter, frac_list, time_limit, table, num_int, count):
    # rho = 10**8
    # alpha0 = 0.4
    epsilon1 = 0.2
    epsilon2 = 0.2
    tol = 0.01
    # alpha = 0.4
    # max_iter = 15
    # frac = 0.8
    # ans = collections.defaultdict(list)
    
    cnt = 0
    obj = 0
    iter = 0
    
    for frac in frac_list:
        ans = collections.defaultdict(list)
        t0_init = time.time()
        res = optimization_un(t0_init,  max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2,  lamd, tol, cnt, obj, iter)
        eval_pip = evaluate_un(res[0], covariates_X, treat_max)
        ans_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in ans.items() ]))
        ans_df["accumulated_time"] = initial_time + ans_df["time"].cumsum()
        table[(num_int, count)][("pip", frac)] = (res, ans_df)
        table[(num_int, count)][("eval_pip", frac)] = eval_pip
    res_global = optimization_global_un(b0_init, b0, score,  lamd, time_limit, count)
    eval_mip = evaluate_un(res_global[0], covariates_X, treat_max)
    table[(num_int, count)]["mip"] = res_global  
    table[(num_int, count)]["eval_mip"] = eval_mip   


    

def run_main_un(count, num_int_list, N_list, num_each_list, time_limit_list,  frac_list):

    lamd = 0.01
    
    max_iter = 10
    for (num_int, num_each, N, time_limit) in zip(num_int_list, num_each_list, N_list, time_limit_list):
        table_count_num_int = collections.defaultdict(dict)
        # num_val_list = np.random.uniform(-1, 1, (num_each, P))
        # data = genData(N, dim, K, P, num_val_list)
        # score = calScoreMulti_est(data)
        # data, score = generate_data(N, dim, K, P, num_val_list)
        data = pd.read_pickle(f"C:/Users/yuefang/Dropbox/policy_learning_inequality/code/data_saved/data_{num_int}_{count}.pickle")
        score = pd.read_pickle(f"C:/Users/yuefang/Dropbox/policy_learning_inequality/code/data_saved/score_{num_int}_{count}.pickle")
        print("read data done")
        treat_max = data[-1]
        covariates_X = data[5]
        del data
        score_unit = score[2]
        initial_score = np.sum(score_unit, axis = 0)
        initial_score = (initial_score - np.min(initial_score)) / (np.max(initial_score) - np.min(initial_score))
        b0_init = initial_score
        print("Start optimization")
        initial_sol = optimization_linear_un(b0_init, score, lamd)
        #global_sol = optimization_global_un(b0_init, None, score, lamd, time_limit, count)
        b0 = initial_sol[0]
        initial_time = initial_sol[-1]
        main_un(b0_init, b0, score, covariates_X, treat_max, initial_time,  lamd, max_iter, frac_list, time_limit, table_count_num_int, num_int, count)
        path = 'C:/Users/yuefang/Dropbox/policy_learning_inequality/code/results_unconstrained_240629'
        os.chdir(path)
        with open(f'table_{count}_{num_int}_maxiter10_alt.pkl', 'wb') as fp:
            pickle.dump(table_count_num_int, fp)

def get_model(X, D, model_type):
    treat_num = len(np.unique(D))
    if model_type == 'logistic':
        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        prob_model = lr.fit(X, D)
       

    elif model_type == 'rf':
        rf = RandomForestClassifier(max_depth=5, random_state=0)
        prob_model = rf.fit(X, D)
       
    elif model_type == 'gbdt':
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        
        prob_model = gb.fit(X, D)
        
    return prob_model


def calScoreMulti_est_un(data):
    Y, df_D, A, prob, df_cov, df_X, df_Z, treat_max = data
    P = df_X.shape[1]
    dim = df_cov.shape[1]
    N = len(Y)
    K = len(np.unique(A))
    M = np.max(Y) - np.min(Y)

    df_DY = pd.DataFrame(df_D*np.array(Y)[:,np.newaxis])
    df_DY = df_DY.add_prefix('DY_')
    df_all = pd.concat([df_cov, df_D, df_DY], axis = 1)
    df_all["Y"] = Y
    df_all["num"] = np.ones(N)
    group_df_D = df_all.groupby(list(df_X.columns)).sum()
    num_val_dim = group_df_D.shape[0] # count the number of unique covariate combinations
    prob_est = np.array(group_df_D[df_D.columns.tolist()])/np.array(group_df_D["num"])[:, np.newaxis]
    print(np.sum(prob_est==0))
    if np.sum(prob_est==0) > 0:
        return None
    prob_est_df = (group_df_D[df_D.columns.tolist()]/np.array(group_df_D["num"])[:, np.newaxis]).reset_index()
    # prob_est_expand = prob_est[:, np.newaxis, :, np.newaxis]
    # prob_est_expand = prob_est_expand * prob_est_expand.transpose((3, 0, 1, 2))

    x =  np.array(prob_est_df.iloc[:,:P]).reshape(num_val_dim, P)    
    score_unit = np.array(group_df_D[df_DY.columns.tolist()]/prob_est)

    # Y_matrix = np.tile(Y, (N, 1))
    # Y_max_neg =  M - np.maximum(Y_matrix, Y_matrix.T)
    # x_pair_dic = collections.defaultdict(dict)
    # for i in range(N):
    #     xi = df_X.iloc[i].tolist()
    #     x_value = tuple(xi+xi)
    #     if (A[i], A[i]) not in x_pair_dic[x_value]:
    #             x_pair_dic[x_value][(A[i], A[i])] = Y_max_neg[i, i]
    #     else:
    #         x_pair_dic[x_value][(A[i], A[i])] += Y_max_neg[i, i]
    #     for j in range(i+1, N):
    #         xj = df_X.iloc[j].tolist()
    #         x_value = tuple(xi+xj)
    #         if (A[i], A[j]) not in x_pair_dic[x_value]:
    #             x_pair_dic[x_value][(A[i], A[j])] = Y_max_neg[i, j]
    #         else:
    #             x_pair_dic[x_value][(A[i], A[j])] += Y_max_neg[i, j]
    #         x_value = tuple(xj+xi)
    #         if (A[j], A[i]) not in x_pair_dic[x_value]:
    #             x_pair_dic[x_value][(A[j], A[i])] = Y_max_neg[j, i]
    #         else:
    #             x_pair_dic[x_value][(A[j], A[i])] += Y_max_neg[j, i]       
    # x_pair_arr = {}
    # for key in x_pair_dic:
    #     if key not in x_pair_arr:
    #         x_pair_arr[key] = np.zeros((K, K))
    #         for item in x_pair_dic[key]:
    #             x_pair_arr[key][item[0]][item[1]] = x_pair_dic[key][item]
    # prob_est_expand_dic = {}
    # for i in range(prob_est_df.shape[0]):
    #     for j in range(prob_est_df.shape[0]):
    #         prob_est_expand_dic[tuple(prob_est_df[df_X.columns.tolist()].values.tolist()[i]+prob_est_df[df_X.columns.tolist()].values.tolist()[j])] = np.array(prob_est_expand[i][j])
    
    # score_pair_dic = {}
    # for key in prob_est_expand_dic:
    #     score_pair_dic[key] = x_pair_arr[key]/prob_est_expand_dic[key]
    # score_pair = np.array(list(score_pair_dic.values())).reshape(num_val_dim, num_val_dim, K, K)
    score_pair = None
    
    return (x, score_pair, score_unit, M, N)

def est_score_alt(Y, X, X_pol, A, model_type = 'rf'):
    num_fold = 5
    model_type = 'rf'
    P = X_pol.shape[1]
    dim = X.shape[1]
    N = len(Y)
    K = len(np.unique(A))
    M = np.max(Y) - np.min(Y)
    df_D = pd.get_dummies(A, prefix='D')
    D = np.array(df_D)
    df_DY = pd.DataFrame(df_D*np.array(Y)[:,np.newaxis])
    df_DY = df_DY.add_prefix('DY_')
    df_all = pd.concat([X, df_D, df_DY], axis = 1)
    df_all["Y"] = Y
    df_all["num"] = np.ones(N)
    # group_df_D = df_all.groupby(list(X_pol.columns)).sum()
    # num_val_dim = group_df_D.shape[0] # count the number of unique covariate combinations
    # prob_est = est_score(num_fold, Y, X, D, model_type)
    cv = KFold(n_splits=num_fold, shuffle=True, random_state=0)
        # Y = data[outcome_var]
        # X = data.drop([outcome_var, treat_var], axis=1)
        # D = data[treat_var]
    treat_num = len(np.unique(A))
    D_pred = np.zeros((len(Y), treat_num))
    Y_pred = np.zeros((len(Y), treat_num))
    cv_idx_dic = {}
    X = np.array(X)
    A = np.array(A)
    A = A.astype(int)

    for i, (train_index, test_index) in enumerate(cv.split(X,A)):
        cv_idx_dic[i] = train_index
        X_train, X_test, A_train, A_test = X[train_index], X[test_index], A[train_index], A[test_index]
        
        prob_model = get_model(X_train, A_train, model_type)
        D_pred[test_index,:] = prob_model.predict_proba(X_test)

    score_unit = np.zeros((len(Y), treat_num))
    # for k in range(treat_num):
    #     score_unit[:,k] = (A == k) / D_pred[:,k] * Y 
    # D_pred_expand = D_pred[:, np.newaxis, :, np.newaxis]
    # D_pred_expand = D_pred_expand * D_pred_expand.transpose((3, 0, 1, 2))
    # Y_matrix = np.tile(Y, (N, 1))
    # Y_max_neg =  M - np.maximum(Y_matrix, Y_matrix.T)

    # expanded_Y_max_neg = np.repeat(Y_max_neg[:, :, np.newaxis, np.newaxis], K, axis=2)
    # expanded_Y_max_neg = np.repeat(expanded_Y_max_neg, K, axis=3)

    score_pair = None
    return (X_pol, score_pair, score_unit, M, N)
def generate_data_alt(N, dim, K, P, num_val_list):
    data = genData(N, dim, K, P, num_val_list)
    score = calScoreMulti_est_un(data)
    if score is None:
        return generate_data_alt(N, dim, K, P, num_val_list)
    else:
        return (data, score)
# def generate_data_alt(N, dim, K, P, num_val_list):
#     data = genData(N, dim, K, P, num_val_list)
#     Y, df_D, A, prob, df_cov, df_X, df_Z, treatment_max = data
#     # score = est_score_alt(Y, df_cov, df_X, A, model_type = 'rf')
    
#     return (data, score)
    