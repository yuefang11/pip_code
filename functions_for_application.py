import numpy as np
import pandas as pd
import math
import time
import gurobipy as gp
import random
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

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
def optimization_linear(b0_init, score, rho, lamd, alpha):
    t0 = time.time()
    x, score_pair, score_unit, M, N = score
    B = 1
    # B_bar = np.min(-B*np.sum(abs(x), axis=1))
    B_bar = -2 * B * np.sum(abs(x), axis=1)[:, np.newaxis] + b0_init - 1 - 0.001
    x_card = score_unit.shape[0]

    K = score_unit.shape[1]
    P = x.shape[1]

    model = gp.Model()
    # model.setParam('Method', 1)
    model.setParam('OutputFlag', 0)
    # model.setParam('NonConvex', 2)
    model.setParam('TimeLimit', 600)
    # model.setParam('MIPFocus', 1)
    # model.setParam('IntegralityFocus', 1)
    # model.setParam('NumericFocus', 3)
    # model.setParam('FeasibilityTol', 1e-9)

    beta = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, lb=-1, ub = 1, name="beta")
    z = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, lb = 0, ub = 1, name="z")
    eta = model.addVars(x_card, x_card, K, K, vtype=gp.GRB.CONTINUOUS, lb = 0, name="eta")
    gamma_var = model.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, name="gamma")
    xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta")
    # max_xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta_max")
    
    model.setObjective(gp.quicksum(z[x_idx, j_idx] * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))/N \
                       - rho * gamma_var, sense = gp.GRB.MAXIMIZE)

    for x_idx in range(x_card):
        for j_idx in range(K):
            model.addConstr(xbeta[x_idx, j_idx] - (gp.quicksum(beta[j_idx, p]*x[x_idx][p] for p in range(P)) + b0_init[j_idx]) == 0)
            
    # for x_idx in range(x_card):
    #     for j_idx in range(K):
    #         # list_xbeta = []
    #         for k in range(K):
    #             if k != j_idx:
    #                 # list_xbeta.append(xbeta[x_idx, k])
    #                 model.addConstr(max_xbeta[x_idx, j_idx] >= xbeta[x_idx, k])
            # model.addConstr(max_xbeta[x_idx, j_idx]  == gp.max_(list_xbeta))
    for x_idx1 in range(x_card):
        for x_idx2 in range(x_card):
            for j_idx1 in range(K):
                for j_idx2 in range(K):
                    # model.addConstr(eta[x_idx1, x_idx2, j_idx1, j_idx2] == gp.min_(z[x_idx1, j_idx1], z[x_idx2, j_idx2]))
                    model.addConstr(eta[x_idx1, x_idx2, j_idx1, j_idx2] <= z[x_idx1, j_idx1])
                    model.addConstr(eta[x_idx1, x_idx2, j_idx1, j_idx2] <= z[x_idx2, j_idx2])       
    for x_idx in range(x_card):
        for j_idx in range(K):
            for k in range(K):
                if k != j_idx:
                    model.addConstr(xbeta[x_idx, j_idx] - xbeta[x_idx, k] - 0.001 >=  B_bar[x_idx, j_idx] * (1 - z[x_idx, j_idx]))
            
    model.addConstr((gp.quicksum(score_pair[x_idx1][x_idx2][j_idx1][j_idx2] * eta[x_idx1, x_idx2, j_idx1, j_idx2]  for x_idx1 in range(x_card) for x_idx2 in range(x_card) for j_idx1 in range(K) for j_idx2 in range(K)) ) / (N**2) \
                    + (1 + (alpha))/N * (gp.quicksum(z[x_idx, j_idx] * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K)) ) + gamma_var >= M)
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
    
    print(obj, obj + rho * gamma_var.x)

    t1 = time.time()
    time_used = t1 - t0
    print(time_used)
    return (b1, obj, time_used)


def optimization_global(b0_init, b0, score, rho, lamd, alpha, time_limit, count):
    t0 = time.time()

    x, score_pair, score_unit, M, N = score
    B = 1
    # B_bar = np.min(-B*np.sum(abs(x), axis=1))
    B_bar = -2 * B * np.sum(abs(x), axis=1)[:, np.newaxis] + b0_init - 1 - 0.001
    x_card = score_unit.shape[0]
    K = score_unit.shape[1]
    P = x.shape[1]
    num_int_global = K * x_card
    # build the gurobi model, set the parameters
    model = gp.Model()
    # model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)
    model.setParam('IntegralityFocus', 1)
    model.setParam('NumericFocus', 3)
    model.setParam('FeasibilityTol', 1e-6)
    model.params.LogFile=f'application/model_mip_{count}_alt.log'
    # add the variables
    beta = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, lb=-1, ub = 1, name="beta")
    z = model.addVars(x_card, K, vtype=gp.GRB.BINARY, name="z")
    eta = model.addVars(x_card, x_card, K, K, vtype=gp.GRB.CONTINUOUS, lb = 0, name="eta")
    gamma_var = model.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, name="gamma")
    xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta")
    # max_xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta_max")
    t = model.addVar(vtype=gp.GRB.CONTINUOUS, name="t")
    s = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, name="s")
    if b0 is not None:
        for k in range(K):
            for p in range(P):
                beta[k,p].Start = b0[k][p]

    # set the objective
    model.setObjective(gp.quicksum(z[x_idx, j_idx] * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))/N \
                       - rho * gamma_var - lamd * t, sense = gp.GRB.MAXIMIZE)
    model.addConstrs((beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstrs((-beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstr(gp.quicksum(s[j, p] for j in range(K) for p in range(P)) == t )
    # set the constraints
    for x_idx in range(x_card):
        for j_idx in range(K):
            model.addConstr(xbeta[x_idx, j_idx] - (gp.quicksum(beta[j_idx, p] * x[x_idx][p] for p in range(P)) + b0_init[j_idx]) == 0)          
    # for x_idx in range(x_card):
    #     for j_idx in range(K):
    #         for k in range(K):
    #             if k != j_idx:
    #                 model.addConstr(max_xbeta[x_idx, j_idx] >= xbeta[x_idx, k])    
    for x_idx1 in range(x_card):
        for x_idx2 in range(x_card):
            for j_idx1 in range(K):
                for j_idx2 in range(K):
                    model.addConstr(eta[x_idx1, x_idx2, j_idx1, j_idx2] <= z[x_idx1, j_idx1])
                    model.addConstr(eta[x_idx1, x_idx2, j_idx1, j_idx2] <= z[x_idx2, j_idx2])
    for x_idx in range(x_card):
        for j_idx in range(K):
            for k in range(K):
                if k != j_idx:
                    model.addConstr(xbeta[x_idx, j_idx] - xbeta[x_idx, k] - 0.001 >=  B_bar[x_idx, j_idx] * (1 - z[x_idx, j_idx]))
            # model.addConstr(xbeta[x_idx, j_idx] - max_xbeta[x_idx, j_idx] - 0.001 >=  B_bar * (1 - z[x_idx, j_idx]))
            
    model.addConstr((gp.quicksum(score_pair[x_idx1][x_idx2][j_idx1][j_idx2] * eta[x_idx1, x_idx2, j_idx1, j_idx2] \
                                  for x_idx1 in range(x_card) for x_idx2 in range(x_card) for j_idx1 in range(K) for j_idx2 in range(K)) ) / (N**2) \
                                    + (1 + (alpha )) * (gp.quicksum(z[x_idx, j_idx] * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))) / N + gamma_var >= M)



    model.update()
    model.optimize()

    # check the objective

    obj = model.objVal
    print(obj, obj + rho * gamma_var.x + lamd * t.x)
    print(gamma_var.x)
    

    # check the constraint
    gini = (-(gp.quicksum(z[x_idx, j_idx].x * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))) / N + M - gamma_var.x-
                    (gp.quicksum(score_pair[x_idx1][x_idx2][j_idx1][j_idx2] * eta[x_idx1, x_idx2, j_idx1, j_idx2].x for x_idx1 in range(x_card) for x_idx2 in range(x_card) for j_idx1 in range(K) for j_idx2 in range(K)) ) / (N**2)
                    ).getValue() / ((gp.quicksum(z[x_idx, j_idx].x * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))) / N).getValue()
    print(gini)
    
    # check the number of non-distinguishable 
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
    return (b1, obj,  obj + rho * gamma_var.x + lamd * t.x, gini, time_used, gamma_var.x)


def optimization(t0_init, relaxed_list, max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2, rho, lamd, alpha, tol, cnt = 0, obj = 0, iter = 0):
    x, score_pair, score_unit, M, N = score
    print(cnt, iter, epsilon1, epsilon2)
    
    B = 1
    # B_bar = np.min(-B*np.sum(abs(x), axis=1))
    B_bar = -2 * B * np.sum(abs(x), axis=1)[:, np.newaxis] + b0_init - 1 - 0.001
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
        return optimization(t0_init, relaxed_list, max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2, rho, lamd, alpha, tol, 0, obj, iter)
    
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

    right_idx_dic = collections.defaultdict(list)
    z_idx_dic = collections.defaultdict(list)
    for i in range(len(right_idx)):
        right_idx_dic[right_idx[i][0]].append(right_idx[i][1])
    for i in range(len(z_idx)):
        z_idx_dic[z_idx[i][0]].append(z_idx[i][1])


    Const_pair = np.sum([score_pair[right_idx[i][0]][right_idx[j][0]][right_idx[i][1]][right_idx[j][1]] for i in range(len(right_idx)) for j in range(len(right_idx))]) 
    Const_unit = np.sum([score_unit[right_idx[i][0]][right_idx[i][1]] for i in range(len(right_idx))])
    t0 = time.time()
    # build the model and set the parameters
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 120)
    # model.setParam('MIPFocus', 1)
    # model.setParam('IntegralityFocus', 1)
    # model.setParam('NumericFocus', 3)
    # model.setParam('FeasibilityTol', 1e-9)
    if obj_init >= 0:
        model.params.BestObjStop = obj_init + 0.1
    
    # add the variables
    beta = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, lb=-1, ub = 1, name="beta")
    z = model.addVars(np.arange(len(z_idx)), vtype=gp.GRB.BINARY, name="z")
    eta = model.addVars(np.arange(len(z_idx)), np.arange(len(z_idx)), vtype=gp.GRB.CONTINUOUS, lb = 0, name="eta")
    gamma_var = model.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, name="gamma")
    xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta")
    # max_xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta_max")
    t = model.addVar(vtype=gp.GRB.CONTINUOUS, name="t")
    s = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, name="s")
    for k in range(K):
        for p in range(P):
            beta[k,p].Start = b0[k][p]
    
    # set the objective
    model.setObjective((gp.quicksum(z[i] * score_unit[z_idx[i][0], z_idx[i][1]] 
                    for i in range(len(z_idx))) +  Const_unit)/N  - rho * gamma_var \
                         - lamd * t, sense = gp.GRB.MAXIMIZE)
    model.addConstrs((beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstrs((-beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstr(gp.quicksum(s[j, p] for j in range(K) for p in range(P)) == t )
    # set the constraints
    for x_idx in range(x_card):
        for j_idx in range(K):
            model.addConstr(xbeta[x_idx, j_idx] - (gp.quicksum(beta[j_idx, p]*x[x_idx][p] for p in range(P)) + b0_init[j_idx]) == 0)
    
    # for x_idx in range(x_card):
    #     for j_idx in range(K):
    #         for k in range(K):
    #             if k != j_idx:
    #                 model.addConstr(max_xbeta[x_idx, j_idx] >= xbeta[x_idx, k])
    for i in range(len(z_idx)):
        for j in range(len(z_idx)):
            model.addConstr(eta[i, j] <= z[i])
            model.addConstr(eta[i, j] <= z[j])
    for i in range(len(z_idx)):
        x_idx = z_idx[i][0]
        j_idx = z_idx[i][1]
        for k in range(K):
            if k != j_idx:
                model.addConstr(xbeta[x_idx, j_idx] - xbeta[x_idx, k] - 0.001 >=  B_bar[x_idx, j_idx] * (1 - z[i]))
        # model.addConstr(xbeta[x_idx, j_idx] - max_xbeta[x_idx, j_idx] - 0.001 >=  B_bar *(1 - z[i]))
    for i in range(len(right_idx)):
        x_idx = right_idx[i][0]
        j_idx = right_idx[i][1]
        for k in range(K):
            if k != j_idx:
                model.addConstr(xbeta[x_idx, j_idx] - xbeta[x_idx, k] - 0.001 >=  0)
        # model.addConstr(xbeta[x_idx, j_idx] - max_xbeta[x_idx, j_idx] - 0.001  >= 0)
    model.addConstr((gp.quicksum(score_pair[z_idx[i][0]][z_idx[j][0]][z_idx[i][1]][z_idx[j][1]]*eta[i, j] for i in range(len(z_idx)) for j in range(len(z_idx)) ) \
                     + 2 * gp.quicksum(score_pair[z_idx[i][0]][right_idx[j][0]][z_idx[i][1]][right_idx[j][1]]*z[i] for i in range(len(z_idx)) for j in range(len(right_idx))) + Const_pair) / (N**2) \
                        + (1 + (alpha )) * (gp.quicksum( z[i] * score_unit[z_idx[i][0], z_idx[i][1]] for i in range(len(z_idx)))  +  Const_unit)/N + gamma_var >= M)
    
    model.update()
    model.optimize()
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
    obj = model.objVal
    print(obj_init, obj, obj + gamma_var.x * rho + lamd * t.x)
    print(t.x)
    gini = (-(gp.quicksum(z[i].x * score_unit[z_idx[i][0], z_idx[i][1]] for i in range(len(z_idx)))  +  Const_unit)/N + M - \
                (gp.quicksum(score_pair[z_idx[i][0]][z_idx[j][0]][z_idx[i][1]][z_idx[j][1]]*eta[i, j].x for i in range(len(z_idx)) for j in range(len(z_idx)) ) \
                + 2 * gp.quicksum(score_pair[z_idx[i][0]][right_idx[j][0]][z_idx[i][1]][right_idx[j][1]]*z[i].x for i in range(len(z_idx)) for j in range(len(right_idx))) + Const_pair) / (N**2) ).getValue() \
                                        / ((gp.quicksum(z[i].x * score_unit[z_idx[i][0], z_idx[i][1]] for i in range(len(z_idx)))  +  Const_unit)/N).getValue()
    
    print(gini)
    # con_trans = (-(gp.quicksum(z[i].x * score_unit[z_idx[i][0], z_idx[i][1]] for i in range(len(z_idx)))  +  Const_unit)/N + M - gamma_var.x - \
    #              (gp.quicksum(score_pair[z_idx[i][0]][z_idx[j][0]][z_idx[i][1]][z_idx[j][1]]*eta[i, j].x for i in range(len(z_idx)) for j in range(len(z_idx)) ) \
    #               + 2 * gp.quicksum(score_pair[z_idx[i][0]][right_idx[j][0]][z_idx[i][1]][right_idx[j][1]]*z[i].x for i in range(len(z_idx)) for j in range(len(right_idx))) + Const_pair) / (N**2) ).getValue() \
    #                                       / ((gp.quicksum(z[i].x * score_unit[z_idx[i][0], z_idx[i][1]] for i in range(len(z_idx)))  +  Const_unit)/N).getValue()
    # print(con_trans)
   
    ans["obj_init"].append(obj_init)
    ans["obj"].append(obj)
    ans["welfare"].append(obj + gamma_var.x * rho + lamd * t.x)
    # gini = con_trans 
    ans["gini"].append(gini)
    # ans["con_trans"].append(con_trans)
    ans["time"].append(t1-t0)
    ans["gamma"].append(gamma_var.x)

    print(time_used)
    cur_time = time.time()
    if iter >= max_iter or cur_time - t0_init > 3600:
        return (b1, obj, obj+gamma_var.x+lamd*t.x, gini, time_used, z_idx, gamma_var.x)
    if abs(obj - obj_init) <= tol:
        b0 = b1
        epsilon1 = epsilon1 * 2
        epsilon2 = epsilon2 * 2
        iter += 1
        return optimization(t0_init, relaxed_list, max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2, rho, lamd, alpha, tol, 0, obj, iter) 
    else:
        b0 = b1
        epsilon1 = epsilon1 * (4/5)
        epsilon2 = epsilon2 * (4/5)
        return optimization(t0_init, relaxed_list, max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2, rho, lamd, alpha, tol, cnt+1, obj, iter)


# table = collections.defaultdict(dict)



def run_code(data, table, idx):    
    
# for idx, data in enumerate(folds):
    Y = data['earnings']
    A = data['treat']
    # dummy_site = pd.get_dummies(data['siteno'], prefix='site')
    # dummy_race = pd.get_dummies(data['race'], prefix='race')
    X = data[['edu', 'prevearn', 'age', 'hascar', 'lookwrk',
       'weekswrk', 'curempl', 'male', 'phonehom', 'siteno_1', 'siteno_2',
       'siteno_3', 'siteno_4', 'siteno_5', 'siteno_6', 'siteno_7', 'siteno_8',
       'siteno_9', 'siteno_10', 'siteno_11', 'siteno_12', 'siteno_13',
       'siteno_14', 'siteno_15', 'siteno_16', 'siteno_17', 'race_1', 'race_2',
       'race_3', 'race_4', 'race_5']]
    # X = pd.concat([X, dummy_site, dummy_race], axis=1)
    X_pol = data[['edu','prevearn']]
    # Delete data to free memory
    
    del data

    num_fold = 5
    model_type = 'rf'
    P = X_pol.shape[1]
    dim = X.shape[1]
    N = len(Y)
    K = len(np.unique(A))

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
    Y = np.array(Y)
    X = np.array(X)
    A = np.array(A)
    A = A.astype(int)


    for i, (train_index, test_index) in enumerate(cv.split(X,A,Y)):
        cv_idx_dic[i] = train_index
        X_train, X_test, A_train, A_test, Y_train, Y_test = X[train_index], X[test_index], A[train_index], A[test_index], Y[train_index], Y[test_index]
        
        prob_model = get_model(X_train, A_train, model_type)
        D_pred[test_index,:] = prob_model.predict_proba(X_test)
        outcome_model = {}
        for j in range(treat_num):
            outcome_model[j] = RandomForestRegressor(random_state=0)
            outcome_model[j].fit(X_train[A_train == j], Y_train[A_train == j])
            Y_pred[test_index, j] = outcome_model[j].predict(X_test)
        # outcome_model = RandomForestRegressor(random_state=0)
        # outcome_model.fit(X_train, Y[train_index])
        # Y_pred[test_index, A_test] = outcome_model.predict(X_test)
    D_pred = np.clip(D_pred, 0.1, 0.9)
    # D_pred = np.tile(np.sum(D, axis=0) / N, N).reshape((N,3))
    score_unit = np.zeros((len(Y), treat_num))


    for k in range(treat_num):
        score_unit[:,k] = (A == k) / D_pred[:,k] * (Y - Y_pred[:,k]) + Y_pred[:, k]

    score_unit = score_unit - np.min(score_unit)
    M = np.max(score_unit)
    expanded_score_unit = np.tile(score_unit[:, :, np.newaxis, np.newaxis], (1, 1, N, K))
    expanded_score_unit = expanded_score_unit.transpose((0, 2, 1, 3))
    score_pair = np.maximum(expanded_score_unit, expanded_score_unit.transpose((1, 0, 2, 3)))
    

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    X_pol_norm = scaler.fit_transform(X_pol)

    # X_pol_norm = np.concatenate([X_pol_norm, np.ones((X_pol_norm.shape[0], 1))], axis=1)
    score = (X_pol_norm, score_pair, score_unit, M, N)
    rho = 10**8
    lamd = 0.01
    alpha0 = 0.3
    max_iter = 10
    initial_score = np.sum(score_unit, axis = 0)
    initial_score = (initial_score - np.min(initial_score)) / (np.max(initial_score) - np.min(initial_score))
    b0_init = initial_score
    print("initial sol")
    initial_sol = optimization_linear(b0_init, score, rho, lamd, alpha0)
    b0 = initial_sol[0]
    initial_time = initial_sol[-1]
    epsilon1 = 0.01
    epsilon2 = 0.01
    tol = 0.01
    # alpha = 0.4
    max_iter = 20
    # frac = 0.8
    # ans = collections.defaultdict(list)
    relaxed_list = []
    cnt = 0
    obj = 0
    iter = 0
    # table = collections.defaultdict(dict)
    alpha_list = [0.3]
    frac_list = [0.4]
    time_limit = 3600
    count = 0
    
    for alpha in alpha_list:
        # res_global = optimization_global(b0_init, b0, score, rho, lamd, alpha, time_limit, idx)
        # table[("mip",alpha)] = res_global
        for frac in frac_list:
            print(alpha, frac)
            ans = collections.defaultdict(list)
            t0_init = time.time()
            res = optimization(t0_init, relaxed_list, max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2, rho, lamd, alpha, tol, cnt, obj, iter)
            ans_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in ans.items() ]))
            ans_df["accumulated_time"] = initial_time + ans_df["time"].cumsum()
            table[("pip", alpha, frac)] = (res, ans_df)
        
    with open(f'application/table_{idx}_alt.pkl', 'wb') as fp:
        pickle.dump(table, fp)