import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import time
from sdv.tabular import GaussianCopula
import math
from numpy.polynomial import Polynomial
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

FIGURES_DIR = PROJECT_DIR + '/output/figures'
TABLES_DIR = PROJECT_DIR + '/output/tables'

# TODO: modify make_payout_df to include premiums as calculated based on the training set.
# 1. figure out how to include positive and negative losses as well
# 4. modify min_CVaR_program to include rhos, separate budget constrant into two, one for each zone,
# make each zone's budget \sum I_z + \rho c_k B, think about how to set budget for each zone?

##### Data Creation #####
def make_multi_zone_logit_data(sigma,n=100,nonlinear=False):
    dim = sigma.shape[0]
    theta = np.random.multivariate_normal(mean=np.zeros(dim),cov=sigma,size=n)

    if nonlinear:
        f = random_polynomial(theta)
    else:
        f = theta
    
    signal_variance = np.var(f)
    noise_variance = signal_variance/10
    epsilon = np.random.multivariate_normal(mean=np.zeros(dim),cov=noise_variance*np.eye(dim),size=n)
    l = 1/(1+np.exp(-(f+epsilon)))
    return l, theta

def random_polynomial(theta):
    coefs = np.random.uniform(-1,1,8)
    coefs[::2] = 0
    p = Polynomial(coefs)
    return p(theta)

##### Optimization Functions #####
def min_CVaR_program(pred_y,train_y,params,include_premium=True):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    eps_p = params['epsilon_p']
    epsilon = params['epsilon']
    c_k = params['c_k']
    budget = params['B']
    Ss = params['S']

    n_samples = train_y.shape[0]
    n_zones = train_y.shape[1]
    S = np.tile(Ss,(n_samples,1)) 
    p = np.ones(n_samples)/n_samples
    A = cp.Variable((n_samples,n_zones))
    B = cp.Variable((n_samples,n_zones))
    t = cp.Variable((n_samples,n_zones))
    t_k = cp.Variable(n_samples)
    alpha = cp.Variable((n_samples,n_zones))
    omega = cp.Variable((n_samples,n_zones))
    pi = cp.Variable((n_samples,n_zones))
    gamma = cp.Variable((n_samples,n_zones))
    gamma_k = cp.Variable(n_samples)
    m = cp.Variable()
    K = cp.Variable()

    constraints = []

    # objective, m >= CVaR(l_z - I_z(theta_z))
    constraints.append(t[0,:] + (1/epsilon)*(p @ gamma) <= m*np.ones(n_zones))

    # CVaR constraints for each zone's loss, gamma^k_z >= l_z - min(a_z \hat{l_z}+b_z, K_z) - t
    if include_premium:
        # constraints.append(gamma >= cp.multiply(S,(train_y + pi - omega)) -t)
        constraints.append(gamma >= train_y + pi - omega -t)
        
    else:
        # constraints.append(gamma >= cp.multiply(S,(train_y - omega)) -t)
        constraints.append(gamma >= train_y - omega -t)

    constraints.append(gamma >= 0)

    # Portfolio Capital Requirement CVaR constraint: CVaR(\sum_z I_z(\theta_z)) <= K^P + \sum_z E[I_z(\theta_Z)]
    # constraints.append(t_k + (1/eps_p)*(p @ gamma_k) <= K + (1/n_samples)*cp.sum(cp.multiply(S,omega)))
    # constraints.append(gamma_k >= cp.sum(cp.multiply(S,alpha),axis=1)-t_k)
    constraints.append(t_k + (1/eps_p)*(p @ gamma_k) <= K + (1/n_samples)*cp.sum(omega))
    constraints.append(gamma_k >= cp.sum(alpha,axis=1)-t_k)
    constraints.append(gamma_k >= 0)
    constraints.append(omega <= cp.multiply(pred_y,A)+B)
    constraints.append(omega <= np.ones((n_samples,n_zones)))
    constraints.append(alpha >= cp.multiply(pred_y,A)+B)
    constraints.append(alpha >= 0)

    # budget constraint
    # constraints.append(budget >= cp.sum(cp.multiply(S,alpha)) + c_k*K)
    constraints.append(budget >= (1/n_samples)*cp.sum(alpha) + c_k*K)

    # Premium Constraints
    if include_premium:
        # constraints.append(pi[0,:] == (1/n_samples)*cp.sum(alpha,axis=0) + (1/cp.sum(S[0,:])*c_k*K))
        # constraints.append(pi[0,:] <= pi_bar)
        constraints.append(pi[0,:] == (1/n_samples)*cp.sum(alpha,axis=0))

    for i in range(n_samples-1):
        constraints.append(A[i,:] == A[i+1,:])
        constraints.append(B[i,:] == B[i+1,:])
        constraints.append(t[i,:] == t[i+1,:])
        if include_premium:
            constraints.append(pi[i,:] == pi[i+1,:])

    objective = cp.Minimize(m)
    problem = cp.Problem(objective,constraints)
    # problem.solve(solver=cp.SCIPY, scipy_options={"method":"highs"})
    problem.solve(solver=cp.GUROBI)
    # print('Req capital: {}'.format(K_p.value))
    return (A.value[0,:], B.value[0,:])

##### Evaluation functions #####
def make_payout_dfs(test_x,test_y,pred_model,strike_vals,a,b,params,baseline_premiums=None,opt_premiums=None):
    n_zones = test_y.shape[1]
    
    bdf = pd.DataFrame()
    odf = pd.DataFrame()
    pred_y = pred_model.predict(test_x)
    for zone in np.arange(n_zones):
        pred_col = 'PredictedLosses{}'.format(zone)
        payout_col = 'Payout{}'.format(zone)
        loss_col = 'Losses{}'.format(zone)
        net_loss_col = 'NetLoss{}'.format(zone)

        bdf[pred_col] = pred_y[:,zone]
        bdf[payout_col] = np.maximum(bdf[pred_col]-strike_vals[zone],0)
        bdf[payout_col] = np.minimum(bdf[payout_col],1)
        bdf[loss_col] = test_y[:,zone]
        bdf[net_loss_col] = bdf[loss_col]-bdf[payout_col]
        if baseline_premiums is not None:
            premium = baseline_premiums[zone]
            bdf[net_loss_col] += premium

        odf[pred_col] = pred_y[:,zone]
        odf[payout_col] = np.minimum(a[zone]*odf[pred_col]+b[zone],1)
        odf[payout_col] = np.maximum(0,odf[payout_col])
        odf[loss_col] = test_y[:,zone]
        odf[net_loss_col] = odf[loss_col]-odf[payout_col]
        if opt_premiums is not None:
            premium = opt_premiums[zone]
            odf[net_loss_col] += premium

    size_0, size_1 = params['S']
    odf['TotalPayout'] = size_0*odf['Payout0'] + size_1*odf['Payout1']
    bdf['TotalPayout'] = size_0*bdf['Payout0'] + size_1*bdf['Payout1']
    return bdf, odf

def calculate_premiums_naive(train_x,train_y,pred_model,strike_vals,a,b,params):
    n_zones = train_y.shape[1]
    
    c_k = params['c_k']
    bdf = pd.DataFrame()
    odf = pd.DataFrame()
    pred_y = pred_model.predict(train_x)
    for zone in np.arange(n_zones):
        pred_col = 'PredictedLosses{}'.format(zone)
        payout_col = 'Payout{}'.format(zone)

        bdf[pred_col] = pred_y[:,zone]
        bdf[payout_col] = np.maximum(bdf[pred_col]-strike_vals[zone],0)
        bdf[payout_col] = np.minimum(bdf[payout_col],1)

        odf[pred_col] = pred_y[:,zone]
        odf[payout_col] = np.minimum(a[zone]*odf[pred_col]+b[zone],1)
        odf[payout_col] = np.maximum(0,odf[payout_col])

    odf['TotalPayout'] = odf['Payout0'] + odf['Payout1']
    bdf['TotalPayout'] = bdf['Payout0'] + bdf['Payout1']

    baseline_loss_cvar = CVaR(bdf,'TotalPayout','TotalPayout',0.01)
    baseline_average_payout = bdf['TotalPayout'].mean()
    baseline_required_capital = baseline_loss_cvar-baseline_average_payout
    baseline_premiums = bdf[['Payout0','Payout1']].mean() + 0.5*c_k*baseline_required_capital

    opt_loss_cvar = CVaR(odf,'TotalPayout','TotalPayout',0.01)
    opt_average_payout = odf['TotalPayout'].mean()
    opt_required_capital = opt_loss_cvar-opt_average_payout
    opt_premiums = odf[['Payout0','Payout1']].mean() + 0.5*c_k*opt_required_capital

    return baseline_premiums, opt_premiums

def determine_strike_values(train_y,eval_y,eval_x,pred_model):
    num_zones = eval_y.shape[1]
    best_strike_vals = []
    best_strike_percentiles = []
    pred_losses = pred_model.predict(eval_x)
    for zone in range(num_zones):
        strike_percentiles = np.arange(0.1,0.35,0.05)
        strike_vals = np.quantile(train_y[:,zone],strike_percentiles)
        strike_performance = {}

        for strike_percentile, strike_val in zip(strike_percentiles,strike_vals):
            strike_percentile = np.around(strike_percentile,2)
            insured_loss = np.maximum(eval_y[:,zone]-strike_val,0)
            payout = np.maximum(pred_losses[:,zone]-strike_val,0).reshape(-1,1)
            loss_share_model = LinearRegression().fit(payout,insured_loss)
            share_explained = loss_share_model.coef_[0]
            strike_performance[(strike_percentile,strike_val)] = share_explained

        best_strike_percentile, best_strike_val = max(strike_performance,key=strike_performance.get)
        best_strike_vals.append(best_strike_val)
        best_strike_percentiles.append(best_strike_percentile)
    return best_strike_vals

def determine_budget_params(pred_y,strike_vals,Ss,c_k=0.05):
    bdf = pd.DataFrame()
    n_zones = pred_y.shape[1]
    for zone in np.arange(n_zones):
        pred_col = 'PredictedLosses{}'.format(zone)
        payout_col = 'Payout{}'.format(zone)
        bdf[pred_col] = pred_y[:,zone]
        bdf[payout_col] = np.maximum(bdf[pred_col]-strike_vals[zone],0)
        bdf[payout_col] = np.minimum(bdf[payout_col],1)

    size_0, size_1 = Ss
    bdf['TotalPayout'] = size_0*bdf['Payout0'] + size_1*bdf['Payout1']
    payout_cvar = CVaR(bdf,'TotalPayout','TotalPayout',0.01)
    average_payout = bdf['TotalPayout'].mean()
    required_capital = payout_cvar-average_payout
    capital_costs = c_k*required_capital
    total_costs = bdf['TotalPayout'].mean() + capital_costs
    return total_costs, average_payout

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

def semi_variance(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    semi_variance = (df.loc[df[loss_col] >= q,outcome_col]-q)**2
    return semi_variance.sum()/(len(semi_variance)-1)

def get_summary_stats(df,Ss,cvar_eps=0.2):
    sdf = {}
    sdf['CVaR0'] = CVaR(df,'Losses0','NetLoss0',cvar_eps)
    sdf['CVaR1'] = CVaR(df,'Losses1','NetLoss1',cvar_eps)
    sdf['$|CVaR_2 - CVaR_1|$'] = np.abs(sdf['CVaR0']-sdf['CVaR1'])
    sdf['Max CVaR'] = np.maximum(sdf['CVaR0'],sdf['CVaR1'])

    sdf['VaR0'] = df['NetLoss0'].quantile(1-cvar_eps)
    sdf['VaR1'] = df['NetLoss1'].quantile(1-cvar_eps)
    sdf['$|VaR_2 - VaR_1|$'] = np.abs(sdf['VaR0']-sdf['VaR1'])
    sdf['Max VaR'] = np.maximum(sdf['VaR0'],sdf['VaR1'])

    sdf['SemiVar0'] = semi_variance(df,'Losses0','NetLoss0',cvar_eps)
    sdf['SemiVar1'] = semi_variance(df,'Losses1','NetLoss1',cvar_eps)
    sdf['Max SemiVar'] = np.maximum(sdf['SemiVar0'],sdf['SemiVar1'])

    sdf['Required Capital'] = (CVaR(df,'TotalPayout','TotalPayout',0.01) - df['TotalPayout'].mean())/np.sum(Ss)  
    sdf['Payout CVaR'] = CVaR(df,'TotalPayout','TotalPayout',0.01)
    sdf['Average Cost'] = (df['TotalPayout'].sum() + 0.15*sdf['Required Capital'])/(np.sum(Ss)*df.shape[0])
    sdf['Payout Cost'] = df['TotalPayout'].sum()
    return(sdf)

def get_summary_stats_bs(df):
    means = {}
    confidence_interval = {}
    cols = ['Max CVaR','Max VaR','Max SemiVar','$|VaR_2 - VaR_1|$','Required Capital','Average Cost']
    for col in cols:
        mean = df[col].mean()
        std_dev = df[col].std()/(math.sqrt(df.shape[0]))
        means[col] = str(np.around(mean,2))
        
        lower_bound = np.around(mean - 1.96*std_dev,2)
        upper_bound = np.around(mean + 1.96*std_dev,2)
        confidence_interval[col] = str([lower_bound,upper_bound])
        
    return(means, confidence_interval)

def make_difference_df(bdf,odf):
    cols = ['Max CVaR','Max VaR','Max SemiVar','$|VaR_2 - VaR_1|$','Required Capital','Average Cost']
    ddf = pd.merge(bdf,odf,left_index=True,right_index=True,suffixes=('_baseline','_opt'))
    for col in cols:
        baseline_col = col + '_baseline'
        opt_col = col + '_opt'
        ddf[col] = (ddf[baseline_col]-ddf[opt_col])
    return ddf

def bootstrap_comparison_df(bdf,odf):
    diff_df = make_difference_df(bdf,odf)
    opt_medians, opt_conf = get_summary_stats_bs(odf)
    baseline_medians, baseline_conf = get_summary_stats_bs(bdf)
    diff_medians, diff_conf = get_summary_stats_bs(diff_df)
    sdf = pd.DataFrame([baseline_medians,baseline_conf,opt_medians,opt_conf,diff_medians,diff_conf])
    sdf['Model'] = ['Baseline','','Opt','','Diff','']
    return sdf[['Model','Max CVaR','Max VaR','Max SemiVar','$|VaR_2 - VaR_1|$','Required Capital','Average Cost']]

def plot_bootstrap_results(sigma,params,bdf,odf,scenario_name):
    fig_filename = FIGURES_DIR + '/Bootstrap/{}.png'.format(scenario_name)
    a = np.zeros(2)
    b = np.zeros(2)

    a[0] = odf['a_1'].median()
    a[1] = odf['a_2'].median()
    b[0] = odf['b_1'].median()
    b[1] = odf['b_2'].median()

    strike_vals = np.zeros(2)
    strike_vals[0] = bdf['s_1'].median()
    strike_vals[1] = bdf['s_2'].median()

    square = 'nonlinear' in scenario_name
    train_y, train_x = make_multi_zone_logit_data(sigma,n=200,square=square)
    test_y, test_x = make_multi_zone_data(beta,mu,sigma,n=100,square=square)
    pred_model = LinearRegression().fit(train_x,train_y)
    ebdf, eodf = make_payout_dfs(test_x,test_y,pred_model,strike_vals,a,b,params)

    fig, axes = plt.subplots(1,2,figsize=(10,6))
    plot_payout_functions(ebdf,eodf,a,b,axes)
    fig.savefig(fig_filename)
    plt.close()

def save_bootstrap_results(bdf,odf,scenario_name):
    table_filename = TABLES_DIR + '/Bootstrap/{}.tex'.format(scenario_name)

    sdf = bootstrap_comparison_df(bdf,odf)
    sdf.to_latex(table_filename,float_format='%.2f',escape=False,index=False)

##### Exploration functions #####
def bootstrap_scenario_exploration(regime,S,include_premium=False,n=1000):
    premium_status = 'premium' if include_premium else 'no_premium'

    # No correlation case
    scenario_name = 'no_corr_{}_{}'.format(regime,premium_status)
    sigma = np.array(np.array([[1,0],[0,1]]))
    cvar_eps=0.2
    params = {'epsilon':cvar_eps,'epsilon_p':0.01,'S':np.array([S,S]),'c_k':0.15}

    baseline_results = []
    opt_results = []
    for i in range(n):
        bdict, odict = run_bootstrap_scenario(sigma,params,regime)
        baseline_results.append(bdict)
        opt_results.append(odict)

    odf = pd.DataFrame(opt_results)
    bdf = pd.DataFrame(baseline_results)
    odf['a diff'] = np.abs(odf['a_1'] - odf['a_2'])
    odf['b diff'] = np.abs(odf['b_1']-odf['b_2'])
    print(bootstrap_comparison_df(bdf,odf))
    plot_bootstrap_results(sigma,params,bdf,odf,scenario_name)
    save_bootstrap_results(bdf,odf,scenario_name)

    # Positive Correlation
    scenario_name = 'pos_corr_{}_{}'.format(regime,premium_status)
    sigma = np.array(np.array([[1,0.9],[0.9,1]]))
    baseline_results = []
    opt_results = []
    for i in range(n):
        bdict, odict = run_bootstrap_scenario(sigma,params,regime)
        baseline_results.append(bdict)
        opt_results.append(odict)

    odf = pd.DataFrame(opt_results)
    bdf = pd.DataFrame(baseline_results)
    print(bootstrap_comparison_df(bdf,odf))
    plot_bootstrap_results(sigma,params,bdf,odf,scenario_name)
    save_bootstrap_results(bdf,odf,scenario_name)

    # Negative Correlation
    scenario_name = 'neg_corr_{}_{}'.format(regime,premium_status)
    sigma = np.array(np.array([[1,0.9],[0.9,1]]))
    baseline_results = []
    opt_results = []
    for i in range(n):
        bdict, odict = run_bootstrap_scenario(sigma,params,regime)
        baseline_results.append(bdict)
        opt_results.append(odict)

    odf = pd.DataFrame(opt_results)
    bdf = pd.DataFrame(baseline_results)
    # print(bootstrap_comparison_df(bdf,odf))
    plot_bootstrap_results(beta,mu,sigma,params,bdf,odf,scenario_name)
    save_bootstrap_results(bdf,odf,scenario_name)

def run_bootstrap_scenario(sigma,params,scenario_name,include_premium=False):
    nonlinear = 'nonlinear' in scenario_name
    y, X = make_multi_zone_logit_data(sigma,n=500,nonlinear=nonlinear)
    train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.33)
    test_x, eval_x, test_y, eval_y =train_test_split(test_x,test_y,test_size=0.33)

    np.quantile(train_y[:,0],(0.1,0.25,0.5,0.75,0.9))
    np.quantile(train_y[:,1],(0.1,0.25,0.5,0.75,0.9))

    pred_model = LinearRegression().fit(train_x,train_y)
    pred_y = pred_model.predict(train_x)
    strike_vals = determine_strike_values(train_y,eval_y,eval_x,pred_model)
    params['B'], params['premium income'] = determine_budget_params(pred_y,strike_vals,params['S'],params['c_k'])

    a,b = np.around(min_CVaR_program(pred_y,train_y,params),2)
    baseline_premiums, opt_premiums = calculate_premiums_naive(train_x,train_y,pred_model,strike_vals,a,b,params)
    if include_premium:
        bdf, odf = make_payout_dfs(test_x,test_y,pred_model,strike_vals,a,b,params,baseline_premiums,opt_premiums)
    else:
        bdf, odf = make_payout_dfs(test_x,test_y,pred_model,strike_vals,a,b,params)
    bdict = get_summary_stats(bdf,params['S'],params['epsilon'])
    odict = get_summary_stats(odf,params['S'],params['epsilon'])
    
    bdict['s_1'], bdict['s_2'] = strike_vals
    odict['a_1'], odict['a_2'] = a
    odict['b_1'], odict['b_2'] = b
    odict['p1'], odict['p2'] = opt_premiums
    bdict['p1'], bdict['p2'] = baseline_premiums
    return(bdict,odict)

def debugging():
    regime = 'linear'
    S = 1
    include_premium=True 
    n=10
    premium_status = 'premium' if include_premium else 'no_premium'

    # No correlation case
    scenario_name = 'no_corr_{}_{}'.format(regime,premium_status)
    sigma = np.array(np.array([[1,0],[0,1]]))
    cvar_eps=0.2
    params = {'epsilon':cvar_eps,'epsilon_p':0.01,'S':np.array([S,S]),'c_k':0.15}

    baseline_results = []
    opt_results = []
    for i in range(n):
        bdict, odict = run_bootstrap_scenario(sigma,params,regime)
        baseline_results.append(bdict)
        opt_results.append(odict)

    odf = pd.DataFrame(opt_results)
    bdf = pd.DataFrame(baseline_results)
    odf['a diff'] = np.abs(odf['a_1'] - odf['a_2'])
    odf['b diff'] = np.abs(odf['b_1']-odf['b_2'])
    print(bootstrap_comparison_df(bdf,odf))

start = time.time()
bootstrap_scenario_exploration('linear',8,True,500)
bootstrap_scenario_exploration('nonlinear',45,True,500)
# bootstrap_scenario_exploration('linear',8,False,100)
# bootstrap_scenario_exploration('nonlinear',45,False,100)
end = time.time()
print(end-start)



