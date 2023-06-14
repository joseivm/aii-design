import pandas as pd
import numpy as np
import math 
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import time
from numpy.polynomial import Polynomial

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

FIGURES_DIR = PROJECT_DIR + '/output/figures'
TABLES_DIR = PROJECT_DIR + '/output/tables'

# TODO: FIGURE OUT WHAT OPTIMAL a, b would be, and see if they would be feasible, i think the 
# bilinear thing is throwing it for a funk. 

def generate_data(T=2,n=50,w=[0.3,0.7],num_bad_years=10):
    critical_thresholds = [2,2,2]
    theta = np.random.normal(loc=3,scale=1,size=(n,T))

    # Y = np.minimum(np.maximum(theta - critical_thresholds,0),1)
    Y = theta
    output = np.dot(Y,w)
    worst_years_idx = np.argsort(output)[-num_bad_years:]
    # worst_years = np.zeros(n)
    # worst_years[worst_years_idx] = 1
    # theta = np.maximum(theta-theta.mean(axis=0),0)
    return theta, worst_years_idx, output

def matching_model(theta,bad_years_idx,params):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z

    eps_k = params['epsilon_K']
    c_k = params['c_k']
    max_premium = params['max_premium']
    budget = params['budget']
    T = params['T']
    f_max = params['f_max']
    f_min = params['f_min']
    w_upper = params['w_upper']
    w_lower = params['w_lower']
    m = params['m']

    Y = theta.shape[0]
    D = theta.shape[1]
    p = np.ones(Y)/Y

    z = cp.Variable(Y,boolean=True)
    A = cp.Variable((T,D),boolean=True)
    delta = cp.Variable((T,D-1))
    alpha = cp.Variable((Y,T))
    omega = cp.Variable((Y,T))
    K = cp.Variable()
    bad_years = np.zeros(Y)
    bad_years[bad_years_idx] = 1
    num_bad_years = len(bad_years_idx)
    v = cp.Variable(num_bad_years-1,boolean=True)

    w = cp.Variable(T)
    a = cp.Variable(T)
    b = cp.Variable(T)
    gamma = cp.Variable(Y)
    t_k = cp.Variable()
    h = {}
    for y in range(Y):
        h[y] = cp.Variable((T,D))
    H = cp.Variable((Y,T))
    M = 10000

    constraints = []

    # Making sure z_y = 1 if there was a payout in year y
    constraints.append(cp.sum(alpha,axis=1) <= M*z)
    constraints.append(cp.sum(omega,axis=1) >= 0.001 -M*(1-z))

    # Payout frequency constraints
    constraints.append((1/Y)*cp.sum(z) <= f_max)
    constraints.append((1/Y)*cp.sum(z) >= f_min)

    # Exclusive assignment constraint
    constraints.append(cp.sum(A,axis=0) <= np.ones(D))
    constraints.append(cp.sum(A,axis=1) >= np.ones(T))

    # Continuity constraints
    for d in range(D-1):
        constraints.append(delta[:,d] >= A[:,d+1] - A[:,d])
        constraints.append(delta[:,d] >= -(A[:,d+1] - A[:,d]))
        
    constraints.append(cp.sum(delta,axis=1) <= 2*np.ones(T))
    constraints.append(A[:,0] + A[:,D-1] <= np.ones(T))

    # alpha definition
    constraints.append(alpha >= H + cp.vstack([b]*Y))
    constraints.append(alpha >= 0)

    # omega definition
    constraints.append(omega <= H + cp.vstack([b]*Y))
    constraints.append(omega <= cp.vstack([w]*Y))

    # h definition
    for y in range(Y):
        constraints.append(h[y] <= (cp.reshape(a,(T,1)) @ cp.reshape(theta[y,:],(1,D))))
        constraints.append(h[y] >= 0)
        constraints.append(h[y] <= M*A)
        constraints.append(h[y] >= (cp.reshape(a,(T,1)) @ cp.reshape(theta[y,:],(1,D))) + M*(A-np.ones((T,D))))
        constraints.append(H[y,:] == cp.sum(h[y],axis=1))

    # a and b constraints
    constraints.append(b <= 0)
    constraints.append(a >= 0)

    # Premium constraints
    constraints.append((1/Y)*cp.sum(alpha)+c_k*K <= max_premium)
    # constraints.append(cp.sum(alpha) <= budget)

    # Portfolio CVaR constraint: CVaR(\sum_z I_z(\theta_z)) <= K^P + \sum_z E[I_z(\theta_Z)]
    constraints.append(t_k + (1/eps_k)*(p @ gamma) <= K + (1/Y)*cp.sum(omega))
    constraints.append(gamma >= cp.sum(alpha,axis=1)-t_k)
    constraints.append(gamma >= 0)

    # Ordinality constraints
    for i in range(num_bad_years-1):
        worst_year = bad_years_idx[i]
        next_worst = bad_years_idx[i+1]
        constraints.append(M*v[i] + cp.sum(omega[worst_year,:]) >= cp.sum(alpha[next_worst,:]))

    constraints.append(cp.sum(v) <= m)


    # w constraints
    constraints.append(cp.sum(w) == 1)
    constraints.append(w >= 0)
    constraints.append(w <= w_upper)
    constraints.append(w >= w_lower)
    # constraints.append(cp.sum(alpha,axis=0) <= w_upper*budget)
    # constraints.append(cp.sum(alpha,axis=0) >= w_lower*budget)
    # constraints.append(cp.sum(omega,axis=0) >= w_lower*budget)

    objective = cp.Maximize(bad_years @ (omega @ np.ones(T)))
    problem = cp.Problem(objective,constraints)
    problem.solve()
    
    result_dict = {'A': A.value, 'z':z.value, 'a':a.value, 'b':b.value,'w':w.value,'alpha':alpha.value,'omega':omega.value}
    return result_dict

# tst = matching_model(theta,bad_years,params)

def get_payouts(results,theta):
    a = results['a']
    b = results['b']
    A = results['A']
    w = results['w']

    eta = np.dot(theta,A.T)
    payouts = np.minimum(np.maximum(eta*a + b,0),w)
    payouts = payouts.sum(axis=1) 
    return payouts

def run_scenario():
params = {'epsilon_K':0.01, 'c_k':0.1, 'max_premium':500,'budget':10000,'f_max':0.4,'f_min':1/7,
            'T':2,'w_upper':np.array([0.35,0.75]),'w_lower':np.array([0.25,0.55]),'m':2}
theta, bad_years, Y = generate_data(n=40,w=[0.3,0.7],num_bad_years=10)
tst = matching_model(theta,bad_years,params)
    get_payouts(tst,theta)
    print(tst['A'])
    theta, b = generate_data()
    


    




