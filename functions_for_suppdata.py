import gurobipy as gp
from gurobipy import GRB
from gurobipy import Model, quicksum 
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt                           
#import seaborn as sns                                     
from sklearn.model_selection import train_test_split      
from sklearn.linear_model import LinearRegression         
from sklearn.neural_network import MLPRegressor           
from sklearn.ensemble import GradientBoostingRegressor    
from sklearn.metrics import mean_squared_error            
from sklearn.model_selection import cross_validate 
from scipy.stats import norm
import collections
import math
import time
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.kernel_ridge import KernelRidge
import collections
import pickle
import os

# def dgp1(n, p, error_type = "uniform"):
#     b1 = 0.5
#     b2 = 0.5
#     b3 = 0.5
#     b4 = 0.5

#     X = np.random.uniform(0, 1, (n, p))
#     gamma = np.ones(p)
#     if error_type == "uniform":
#         error0 = np.random.uniform(-1, 1, n)
#         error1 = np.random.uniform(-1, 1, n)
#         error2 = np.random.uniform(-1, 1, n)
#         error3 = np.random.uniform(-1, 1, n)

        
#     elif error_type == "normal":
#         error0 = np.random.normal(0, 1, n)
#         error1 = np.random.normal(0, 1, n)
#         error2 = np.random.normal(0, 1, n)
#         error3 = np.random.normal(0, 1, n)

#     Y0 = np.dot(X, gamma)  + error0
#     Y1 = np.dot(X, gamma) + X[:, 0]*b1 + error1
#     Y2 = np.dot(X, gamma) + X[:, 1]*X[:, 0]*b2 - X[:, 3]**3*b4 + error2
#     Y3 = np.dot(X, gamma) + X[:, 2]*b3 + error3
#     return X, Y0, Y1, Y2, Y3

# def dgp2(n, p, error_type = "uniform"):
#     gamma = np.ones(p)

#     b1 = np.random.uniform(0, 1, n)
#     b2 = np.random.uniform(0, 1, n)
#     b3 = np.random.uniform(0, 1, n) 
#     b4 = np.random.uniform(0, 1, n)

#     X = np.random.uniform(0, 1, (n, p))
#     if error_type == "uniform":
#         error0 = np.random.uniform(-1, 1, n)
#         error1 = np.random.uniform(-1, 1, n)
#         error2 = np.random.uniform(-1, 1, n)
#         error3 = np.random.uniform(-1, 1, n)

        
#     elif error_type == "normal":
#         error0 = np.random.normal(0, 1, n)
#         error1 = np.random.normal(0, 1, n)
#         error2 = np.random.normal(0, 1, n)
#         error3 = np.random.normal(0, 1, n)

#     Y0 = np.dot(X, gamma)  + error0
#     Y1 = np.dot(X, gamma) + X[:, 0]*b1 + error1
#     Y2 = np.dot(X, gamma) + X[:, 1]*X[:, 0]*b2 - X[:, 3]**3*b4 + error2
#     Y3 = np.dot(X, gamma) + X[:, 2]*b3 + error3
#     return X, Y0, Y1, Y2, Y3

# def dgp3(n, p, error_type = "uniform"):
#     gamma = np.ones(p)

#     tau = np.random.uniform(0, 1, n)

#     X = np.random.uniform(0, 1, (n, p))
#     if error_type == "uniform":
#         error0 = np.random.uniform(-1, 1, n)
#         error1 = np.random.uniform(-1, 1, n)
#         error2 = np.random.uniform(-1, 1, n)
#         error3 = np.random.uniform(-1, 1, n)

        
#     elif error_type == "normal":
#         error0 = np.random.normal(0, 1, n)
#         error1 = np.random.normal(0, 1, n)
#         error2 = np.random.normal(0, 1, n)
#         error3 = np.random.normal(0, 1, n)

#     Y0 = np.dot(X, gamma) + error0
#     Y1 = np.dot(X, gamma) + tau * (X[:, 0] + X[:, 1] <= 0.5)+ error1
#     Y2 = np.dot(X, gamma) + tau * (X[:, 1] + X[:, 0]> 0.5) * (X[:, 1] + X[:, 0] <= 1) + error2
#     Y3 = np.dot(X, gamma) + tau * (X[:, 1] + X[:, 0]> 1) + error3
#     return X, Y0, Y1, Y2, Y3


# def dgp4(n, p, error_type = "uniform"):
#     X = np.random.uniform(0, 1, (n, p))
#     if error_type == "uniform":
#         error1 = np.random.uniform(-80, 80, n)
#         error2 = np.random.uniform(-80, 80, n)
#         error3 = np.random.uniform(-80, 80, n)
#         error4 = np.random.uniform(-80, 80, n)
#     elif error_type == "normal":
#         error1 = np.random.normal(0, 40, n)
#         error2 = np.random.normal(0, 40, n)
#         error3 = np.random.normal(0, 40, n)
#         error4 = np.random.normal(0, 40, n)

#     Y1 = 50 * (- X[:, 0] - X[:, 1]  + 0.5 + 0.5 * (X[:, 2] + X[:, 3] - X[:, 4]**2)) + error1
#     Y2 = 50 * (0.5 * (X[:, 2] + X[:, 3] - X[:, 4]**2)) + error2
#     Y3 = 50 * (2 * X[:, 1] -  X[:, 0] - 1 + 0.5 * (X[:, 2] + X[:, 3] - X[:, 4]**2)) + error3
#     Y4 = 50 * ( X[:, 1]- 0.5 + 0.5 * (X[:, 2] + X[:, 3] - X[:, 4]**2)) + error4
#     return X, Y1, Y2, Y3, Y4

# def dgp5(n, p, error_type = "uniform"):
#     X = np.random.uniform(0, 1, (n, p))
#     if error_type == "uniform":
#         error1 = np.random.uniform(-80, 80, n)
#         error2 = np.random.uniform(-80, 80, n)
#         error3 = np.random.uniform(-80, 80, n)
#         error4 = np.random.uniform(-80, 80, n)
#     elif error_type == "normal":
#         error1 = np.random.normal(0, 40, n)
#         error2 = np.random.normal(0, 40, n)
#         error3 = np.random.normal(0, 40, n)
#         error4 = np.random.normal(0, 40, n)

#     Y1 = 50 * (- X[:, 0] - X[:, 1] + X[:, 0] * X[:, 1] + 0.5 + 0.5 * (X[:, 2] + X[:, 3] - X[:, 4]**2)) + error1
#     Y2 = 50 * (0.5 * (X[:, 2] + X[:, 3] - X[:, 4]**2)) + error2
#     Y3 = 50 * (2 * X[:, 1] -  X[:, 0] - 0.5 * X[:, 1]**2 - 0.5 + 0.5 * (X[:, 2] + X[:, 3] - X[:, 4]**2)) + error3
#     Y4 = 50 * ( X[:, 1]- 0.5*X[:, 0]**4 - 0.5 + 0.5 * (X[:, 2] + X[:, 3] - X[:, 4]**2)) + error4
#     return X, Y1, Y2, Y3, Y4



def dgp_1(N, num_each, P, K, error="uniform"):
    gamma = np.ones(P) * 0.001  # Increased base effect significantly
    num_val_list = np.random.uniform(0, 100, (num_each, P)) 
    covariates = num_val_list[np.random.choice(num_val_list.shape[0], N)]
    if error == "uniform":
        epsilon1 = np.random.lognormal(0, 0.5, N)   # Centered lognormal with mean 0
        epsilon2 = np.random.lognormal(0, 0.5, N) 
        epsilon3 = np.random.lognormal(0, 0.5, N)
        epsilon4 = np.random.lognormal(0, 0.5, N)
    elif error == "normal":
        epsilon1 = np.random.normal(0, 1, N)  # Scaled down errors
        epsilon2 = np.random.normal(0, 1, N)
        epsilon3 = np.random.normal(0, 1, N)
        epsilon4 = np.random.normal(0, 1, N)
    # Y1 = np.dot(covariates, gamma) + 0.002*covariates[:, 0]**3 + epsilon1
    # Y2 = np.dot(covariates, gamma) + 0.004*covariates[:, 0] + 0.002*covariates[:, 1]**3 + epsilon2 
    # Y3 = np.dot(covariates, gamma) + 0.003*covariates[:, 1] + 0.002*covariates[:, 2]**3 + epsilon3 
    # Y4 = np.dot(covariates, gamma) + 0.005*covariates[:, 3] + 0.004*covariates[:, 4]**3 + epsilon4
    Y1 = np.dot(covariates, gamma) + 0.001*covariates[:, 0]**3 + epsilon1   # Scaled down coefficients and offset
    Y2 = np.dot(covariates, gamma) + 0.002*covariates[:, 0] + 0.001*covariates[:, 1]**3 + epsilon2 
    Y3 = np.dot(covariates, gamma) + 0.001*covariates[:, 1] + 0.001*covariates[:, 2]**3 + epsilon3 
    Y4 = np.dot(covariates, gamma) + 0.002*covariates[:, 3] + 0.001*covariates[:, 4]**3 + epsilon4 
    Y_true = np.vstack((Y1, Y2, Y3, Y4)).T
    prob = np.tile(np.ones(K)*1/K, (N,1))
    D = np.zeros((N, K))
    for i in range(N):
        D[i] = np.random.multinomial(1, prob[i])
    A = np.where(D==1)[1]
    Y = Y1 * (A==0) + Y2 * (A==1) + Y3 * (A==2) + Y4 * (A==3)
    return (Y, Y_true, D, A, covariates)

def dgp_2(N, num_each, P, K, error="uniform"):
    gamma = np.ones(P) * 0.001  # Scaled down base effect further
    b1 = np.random.uniform(0, 0.002, N)  # Scaled down coefficients further
    b2 = np.random.uniform(0, 0.002, N)
    b3 = np.random.uniform(0, 0.002, N)
    b4 = np.random.uniform(0, 0.002, N)
    b12 = np.random.uniform(0, 0.001, N)  # Scaled down coefficients further
    b22 = np.random.uniform(0, 0.001, N)
    b32 = np.random.uniform(0, 0.001, N)
    b42 = np.random.uniform(0, 0.001, N)
    num_val_list = np.random.uniform(0, 100, (num_each, P))
    covariates = num_val_list[np.random.choice(num_val_list.shape[0], N)]
    if error == "uniform":
        epsilon1 = np.random.lognormal(0, 0.5, N)  # Centered lognormal with mean 0
        epsilon2 = np.random.lognormal(0, 0.5, N) 
        epsilon3 = np.random.lognormal(0, 0.5, N) 
        epsilon4 = np.random.lognormal(0, 0.5, N) 
    elif error == "normal":
        epsilon1 = np.random.normal(0, 1, N)  # Scaled down errors further
        epsilon2 = np.random.normal(0, 1, N)
        epsilon3 = np.random.normal(0, 1, N)
        epsilon4 = np.random.normal(0, 1, N)
    Y1 = np.dot(covariates, gamma) + b1*covariates[:, 0]**3 + epsilon1   # Scaled down offset further
    Y2 = np.dot(covariates, gamma) + b2*covariates[:,0] + b22*covariates[:, 1]**3 + epsilon2 
    Y3 = np.dot(covariates, gamma) + b3*covariates[:,1] + b32*covariates[:,2]**3 + epsilon3 
    Y4 = np.dot(covariates, gamma) + b4*covariates[:,3] + b42*covariates[:, 4]**3 + epsilon4 
    Y_true = np.vstack((Y1, Y2, Y3, Y4)).T
    prob = np.tile(np.ones(K)*1/K, (N,1))
    D = np.zeros((N, K))
    for i in range(N):
        D[i] = np.random.multinomial(1, prob[i])
    A = np.where(D==1)[1]
    Y = Y1 * (A==0) + Y2 * (A==1) + Y3 * (A==2) + Y4 * (A==3)
    return (Y, Y_true, D, A, covariates)

# def genData(N, dim, K, P, num_val_list):
#     covariate_primary = num_val_list[np.random.choice(num_val_list.shape[0], N)]
#     covariate_secondary = np.random.uniform(0, 1, size=(N, dim - P))
#     covariate_values = np.hstack((covariate_primary, covariate_secondary))
#     X = covariate_values
#     df_X = pd.DataFrame(covariate_primary)
#     df_X = df_X.add_prefix('X_')
#     df_Z = pd.DataFrame(covariate_secondary)
#     df_Z = df_Z.add_prefix('Z_')
#     df_cov = pd.concat([df_X, df_Z], axis=1)
#     # for i in range(covariate_values.shape[1]):
#     #     covariate_values[:,i] = (covariate_values[:,i] - np.min(covariate_values[:,i])) / ( np.max(covariate_values[:,i]) -  np.min(covariate_values[:,i]))
#     prob = np.tile(np.ones(K)*1/K, (N,1))
#     D = np.zeros((N, K))
#     for i in range(N):
#         D[i] = np.random.multinomial(1, prob[i])
#     A = np.where(D==1)[1]
#     df_D = pd.DataFrame(D)
#     df_D = df_D.add_prefix('D_')
#     # effect = np.zeros(N, K)
#     # for k in range(K):
#     #     coef = np.random.uniform(-2, 2, size=(dim, 1))
#     const = random.choice(df_X.columns.tolist())
#     mu = df_X[const] + np.exp(2 + 0.2*df_X['X_0'] - 0.1*df_X['X_1'] + 2 * df_X['X_0'] * df_X['X_1'] \
#                               + (A == 0)*(-0.8 +  1.8 * df_X['X_1'] - 0.2 * df_X['X_2'])  \
#                                 + (A == 1) * (-1 + 2.1 * df_X['X_1'] - 1.2 * df_X['X_0']) \
#                                     + (A == 2) * (-0.8 + 1.3 * df_X['X_0'] * df_X['X_2']) \
#                                         + (A == 3) * (-0.4 + 1.8 * df_X['X_0'] - 1.2 * df_X['X_1'] * df_X['X_2']))
#     treatment_score = np.zeros((N, K))
#     treatment_score[:, 0] = -0.8 +  1.8 * df_X['X_1'] - 0.2 * df_X['X_2']
#     treatment_score[:, 1] = -1 + 2.1 * df_X['X_1'] - 1.2 * df_X['X_0']
#     treatment_score[:, 2] = -0.8 + 1.3 * df_X['X_0'] * df_X['X_2']
#     treatment_score[:, 3] = -0.4 + 1.8 * df_X['X_0'] - 1.2 * df_X['X_1'] * df_X['X_2']
#     treatment_max = np.argmax(treatment_score, axis=1)  
#     epsilon = np.random.lognormal(0, 0.001, N)
#     Y = mu + epsilon
    
#     return (Y, df_D, A, prob, df_cov, df_X, df_Z, treatment_max)


def calScoreMulti_est(data):
    Y, Y_true, D, A, covariates = data
    N, P = covariates.shape
    K = len(np.unique(A))
    M = np.max(Y) - np.min(Y)
    df_D = pd.DataFrame(D)
    df_D = df_D.add_prefix('D_')
    df_X = pd.DataFrame(covariates)
    df_DY = pd.DataFrame(df_D*np.array(Y)[:,np.newaxis])
    df_DY = df_DY.add_prefix('DY_')
    df_all = pd.concat([df_X, df_D,df_DY], axis = 1)
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

    Y_matrix = np.tile(Y, (N, 1))
    Y_max_neg =  M - np.maximum(Y_matrix, Y_matrix.T)
    x_pair_dic = collections.defaultdict(dict)
    for i in range(N):
        xi = df_X.iloc[i].tolist()
        x_value = tuple(xi+xi)
        if (A[i], A[i]) not in x_pair_dic[x_value]:
                x_pair_dic[x_value][(A[i], A[i])] = Y_max_neg[i, i]
        else:
            x_pair_dic[x_value][(A[i], A[i])] += Y_max_neg[i, i]
        for j in range(i+1, N):
            xj = df_X.iloc[j].tolist()
            x_value = tuple(xi+xj)
            if (A[i], A[j]) not in x_pair_dic[x_value]:
                x_pair_dic[x_value][(A[i], A[j])] = Y_max_neg[i, j]
            else:
                x_pair_dic[x_value][(A[i], A[j])] += Y_max_neg[i, j]
            x_value = tuple(xj+xi)
            if (A[j], A[i]) not in x_pair_dic[x_value]:
                x_pair_dic[x_value][(A[j], A[i])] = Y_max_neg[j, i]
            else:
                x_pair_dic[x_value][(A[j], A[i])] += Y_max_neg[j, i]       
    x_pair_arr = {}
    for key in x_pair_dic:
        if key not in x_pair_arr:
            x_pair_arr[key] = np.zeros((K, K))
            for item in x_pair_dic[key]:
                x_pair_arr[key][item[0]][item[1]] = x_pair_dic[key][item]
    prob_est_expand_dic = {}
    for i in range(prob_est_df.shape[0]):
        for j in range(prob_est_df.shape[0]):
            prob_est_expand_dic[tuple(prob_est_df[df_X.columns.tolist()].values.tolist()[i]+prob_est_df[df_X.columns.tolist()].values.tolist()[j])] = np.array(prob_est_expand[i][j])
    
    score_pair_dic = {}
    for key in prob_est_expand_dic:
        score_pair_dic[key] = x_pair_arr[key]/prob_est_expand_dic[key]
    score_pair = np.array(list(score_pair_dic.values())).reshape(num_val_dim, num_val_dim, K, K)
    
    
    return (x, score_pair, score_unit, M, N)


def calScoreMulti(data):
    Y, X, D, A, prob, X_primary, treat_max = data
    P = X_primary.shape[1]
    N = len(Y)
    K = len(np.unique(A))
    M = np.max(Y) - np.min(Y)
    
    df = pd.DataFrame(X)
    df = df.add_prefix('X_')
    
    df_D = pd.DataFrame(D)
    df_D = df_D.add_prefix('D_')
    df_D = pd.concat([df[['X_0', 'X_1', 'X_2']], df_D], axis = 1)
    df_D["num"] = np.ones(N)
    group_df_D = df_D.groupby(['X_0', 'X_1', 'X_2']).sum()
    prob_est = np.array(group_df_D[["D_0","D_1","D_2","D_3"]]) / np.array(group_df_D["num"])[:, np.newaxis]


    gamma = np.zeros((N, K))
    gammaY = np.zeros((N, K))
    for i in range(N):
        gamma[i] = D[i] / prob[i]
        gammaY[i] = gamma[i] * (Y[i])
    Y_matrix = np.tile(Y, (N, 1))
    Y_max_neg =  M - np.maximum(Y_matrix, Y_matrix.T)


    df["Y"] = Y
    df["A"] = A
    df_score = pd.DataFrame(gamma)
    df_score = df_score.add_prefix('gamma_')
    df_scoreY = pd.DataFrame(gammaY)
    df_scoreY = df_scoreY.add_prefix('gammaY_')
    df_full = pd.concat([df[['X_0', 'X_1', 'X_2']], df_scoreY], axis=1)
    group_df = df_full.groupby(['X_0', 'X_1', 'X_2']).sum()
    group_df = group_df.reset_index()
    
    df_pair = df[['X_0','X_1','X_2']].merge(df[['X_0', 'X_1', 'X_2']], suffixes=('_i', '_j'), how='cross')
    df_pair["Y_max_neg"] = Y_max_neg.flatten()

    
    reshaped_matrix = gamma[:, np.newaxis, :, np.newaxis]

    # Perform the cross product to obtain the N^2*K^2 matrix
    cross_product_matrix = reshaped_matrix * reshaped_matrix.transpose((3, 0, 1, 2))

    # Reshape the result back to n^2*m^2 matrix
    result_matrix = cross_product_matrix.reshape(N**2, K**2)

    result_matrix_Y = result_matrix * np.array(df_pair["Y_max_neg"]).reshape(N**2, 1)
    df_pair_score = pd.DataFrame(result_matrix_Y, columns = [str((i,j)) for i in range(K) for j in range(K)])
    df_pair_score = df_pair_score.add_prefix('gammaY_ij_')
    df_pair_full = pd.concat([df_pair, df_pair_score], axis = 1)
    group_df_pair = df_pair_full.groupby(['X_0_i', 'X_1_i', 'X_2_i', 'X_0_j', 'X_1_j', 'X_2_j']).sum().iloc[:, 1:]
    group_df_pair = group_df_pair.reset_index()
    dim = np.unique(X_primary, axis = 0).shape[0]
    x = np.array(group_df.iloc[:,:P]).reshape(dim, P)
    score_pair = np.array(group_df_pair.iloc[:, 2*P:]).reshape(dim, dim, K, K)
    score_unit = np.array(group_df.iloc[:, P: ])
    return (x, score_pair, score_unit, M, N)


def optimization_global(b0_init, b0, score, rho, lamd, alpha, time_limit,count,error):
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
    # model.setParam('MIPFocus', 1)
    # model.setParam('IntegralityFocus', 1)
    # model.setParam('NumericFocus', 3)
    # model.setParam('FeasibilityTol', 1e-9)
    log_file = f'model_{count}_{num_int_global}_{error}.log'
    # if os.path.exists(log_file):
    #     os.remove(log_file)
    model.params.LogFile=log_file
    # Remove existing log file if it exists
    
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


    # print(con_trans)
    # gini = con_trans 
    # if gini < 0:
    #     if count == 1:
    #         M = 52000
    #     else:
    #         M = 1200
    #     gini = (-(gp.quicksum(z[x_idx, j_idx].x * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))) / N + M -
    #             (gp.quicksum(score_pair[x_idx1][x_idx2][j_idx1][j_idx2] * eta[x_idx1, x_idx2, j_idx1, j_idx2].x for x_idx1 in range(x_card) for x_idx2 in range(x_card) for j_idx1 in range(K) for j_idx2 in range(K)) ) / (N**2)
    #             ).getValue() / ((gp.quicksum(z[x_idx, j_idx].x * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))) / N).getValue()
    # print(gini)
    
    # check the number of non-distinguishable 
    supp = []
    for x_idx in range(x_card):
        supp.append(gp.quicksum(z[x_idx, j_idx].x for j_idx in range(K)).getValue())
    supp = np.array(supp)
    print(np.sum(supp==0), np.sum(np.abs(supp-1)<0.001), np.sum(supp-1>=0.001))
    
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
    model.setParam('OutputFlag', 0)
    # model.setParam('NonConvex', 2)
    model.setParam('TimeLimit', 600)

    beta = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, lb=-1, ub = 1, name="beta")
    z = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, lb = 0, ub = 1, name="z")
    eta = model.addVars(x_card, x_card, K, K, vtype=gp.GRB.CONTINUOUS, lb = 0, name="eta")
    gamma_var = model.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, name="gamma")
    xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta")
    # max_xbeta = model.addVars(x_card, K, vtype=gp.GRB.CONTINUOUS, name="xbeta_max")
    t = model.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, name="t")
    s = model.addVars(K, P, vtype=gp.GRB.CONTINUOUS, name="s")
    

    model.setObjective(gp.quicksum(z[x_idx, j_idx] * score_unit[x_idx, j_idx] for x_idx in range(x_card) for j_idx in range(K))/N \
                       - rho * gamma_var - lamd*t, sense = gp.GRB.MAXIMIZE)
    model.addConstrs((beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstrs((-beta[j, p] <= s[j, p] for j in range(K) for p in range(P)))
    model.addConstr(gp.quicksum(s[j, p] for j in range(K) for p in range(P)) == t )
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
                    model.addConstr(xbeta[x_idx, j_idx] - xbeta[x_idx, k] - 0.001 >=  B_bar[x_idx, j_idx]* (1 - z[x_idx, j_idx]))
            
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
    
    print(obj, obj + rho * gamma_var.x + lamd * t.x)
    print(t.x)    
    t1 = time.time()
    time_used = t1 - t0
    print(time_used)
    return (b1, obj, time_used)



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
        epsilon1 = epsilon1 * 5/6
        epsilon2 = epsilon2 * 5/6
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
        model.params.BestObjStop = obj_init + 10
    
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
    
    # gini = 0
    # if gini < 0:
    #     if dgp==1:
    #         M = 52000
    #     else:
    #         M = 1400
    #     gini =  (-(gp.quicksum(z[i].x * score_unit[z_idx[i][0], z_idx[i][1]] for i in range(len(z_idx)))  +  Const_unit)/N + M - \
    #             (gp.quicksum(score_pair[z_idx[i][0]][z_idx[j][0]][z_idx[i][1]][z_idx[j][1]]*eta[i, j].x for i in range(len(z_idx)) for j in range(len(z_idx)) ) \
    #             + 2 * gp.quicksum(score_pair[z_idx[i][0]][right_idx[j][0]][z_idx[i][1]][right_idx[j][1]]*z[i].x for i in range(len(z_idx)) for j in range(len(right_idx))) + Const_pair) / (N**2) ).getValue() \
    #                                     / ((gp.quicksum(z[i].x * score_unit[z_idx[i][0], z_idx[i][1]] for i in range(len(z_idx)))  +  Const_unit)/N).getValue()
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


def evaluate(b1, covariates_X, treat_max):
    xbeta_res = np.dot(covariates_X, b1.T)
    f_score = np.zeros(xbeta_res.shape)
    for i in range(xbeta_res.shape[1]):
        f_score[:,i] = xbeta_res[:,i] - np.max(np.delete(xbeta_res, i, axis=1), axis = 1) 
    treat_opt = np.argmax(f_score, axis = 1)
    correct_assign = np.mean(treat_opt == treat_max)
    return correct_assign

def main(b0_init, b0, score, covariates_X, treat_max, initial_time, rho, lamd, alpha, max_iter, frac_list, time_limit, table, num_int, count,error):
    # rho = 10**8
    # alpha0 = 0.4
    epsilon1 = 0.01
    epsilon2 = 0.01
    tol = 0.01
    # alpha = 0.4
    # max_iter = 15
    # frac = 0.8
    # ans = collections.defaultdict(list)
    relaxed_list = []
    cnt = 0
    obj = 0
    iter = 0
    res_global = optimization_global(b0_init, b0, score, rho, lamd, alpha, time_limit, count,error)
    eval_mip = evaluate(res_global[0], covariates_X, treat_max)
    for frac in frac_list:
        ans = collections.defaultdict(list)
        t0_init = time.time()
        res = optimization(t0_init, relaxed_list, max_iter, frac, ans, b0_init, b0, score, epsilon1, epsilon2, rho, lamd, alpha, tol, cnt, obj, iter)
        eval_pip = evaluate(res[0], covariates_X, treat_max)
        ans_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in ans.items() ]))
        ans_df["accumulated_time"] = initial_time + ans_df["time"].cumsum()
        table[(num_int, count, alpha, error)]["mip"] = res_global
        table[(num_int, count, alpha, error)][("pip", frac)] = (res, ans_df)
        table[(num_int, count, alpha, error)]["eval_mip"] = eval_mip
        table[(num_int, count, alpha, error)][("eval_pip", frac)] = eval_pip
def run_main(count, num_int_list, N_list, num_each_list, time_limit_list, alpha_list, frac_list):
    # K = 4
    # P = 20
    # dim = 30
    rho = 10**8
    lamd = 0.01
    alpha0 = 0.4
    max_iter = 10
    for (num_int, num_each, N, time_limit) in zip(num_int_list, num_each_list, N_list, time_limit_list):
        for error in ['normal', 'uniform']:
            table_count_num_int = collections.defaultdict(dict)
            # num_val_list = np.random.uniform(-1, 1, (num_each, P))
            # data = genData(N, dim, K, P, num_val_list)
            # score = calScoreMulti_est(data)
            # data, score = generate_data(N, dim, K, P, num_val_list)
            # data = pd.read_pickle(f"C:/Users/yuefang/Dropbox/policy_learning_inequality/code/data_saved/data_{num_int}_{count}.pickle")
            # score = pd.read_pickle(f"C:/Users/yuefang/Dropbox/policy_learning_inequality/code/data_saved/score_{num_int}_{count}.pickle")
            data = pd.read_pickle(f"C:/Users/yuefang/Dropbox/policy_learning_inequality/code/data_supp_250105/data{count}_{error}_{num_int}.pickle")
            score = pd.read_pickle(f"C:/Users/yuefang/Dropbox/policy_learning_inequality/code/data_supp_250105/score{count}_{error}_{num_int}.pickle")
            print("read data done")
            
            Y_true = data[1]
            treat_max = np.argmax(Y_true, axis=1) 
            covariates_X = data[-1]
            del data
            score_unit = score[2]
            initial_score = np.sum(score_unit, axis = 0)
            initial_score = (initial_score - np.min(initial_score)) / (np.max(initial_score) - np.min(initial_score))
            b0_init = initial_score
            print("Start optimization")
            initial_sol = optimization_linear(b0_init, score, rho, lamd, alpha0)
            b0 = initial_sol[0]
            initial_time = initial_sol[-1]
            
            for alpha in alpha_list:
                print(num_int, num_each, N, count, alpha)
                main(b0_init, b0, score, covariates_X, treat_max, initial_time, rho, lamd, alpha, max_iter, frac_list, time_limit, table_count_num_int, num_int, count,error)
            path = 'C:/Users/yuefang/Dropbox/policy_learning_inequality/code/results_supplement_250107'
            os.chdir(path)
            with open(f'table{count}_{error}_{num_int}.pkl', 'wb') as fp:
                pickle.dump(table_count_num_int, fp)


