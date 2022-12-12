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
def make_multi_zone_logit_data(sigma,n=100,nonlinear=False,seed=1):
    dim = sigma.shape[0]
    rng = np.random.default_rng(seed)

    if nonlinear:
        p = random_polynomial(rng)
        loc = get_relevant_domain(p)
        theta = rng.multivariate_normal(mean=np.array([loc,loc]),cov=sigma,size=n)
        f = p(theta)
        signal_variance = np.var(f)
        noise_variance = signal_variance/2000000
        # epsilon = 0
        epsilon = rng.multivariate_normal(mean=np.zeros(dim),cov=noise_variance*np.eye(dim),size=n)
    else:
        theta = rng.multivariate_normal(mean=np.zeros(dim),cov=sigma,size=n)
        f = theta
        signal_variance = np.var(f)
        noise_variance = signal_variance/10
        epsilon = rng.multivariate_normal(mean=np.zeros(dim),cov=noise_variance*np.eye(dim),size=n)

    l = 1/(1+np.exp(-(f+epsilon)))
    return l, theta

def get_relevant_domain(p):
    rng = np.random.default_rng()
    x = np.linspace(-2,2,200)
    f = p(x)
    signal_variance = np.var(f)
    noise_variance = signal_variance/1000
    epsilon = rng.normal(loc=0,scale=math.sqrt(noise_variance))
    epsilon = 0
    y = 1/(1+np.exp(-(f+epsilon)))
    min_idx = (np.abs(y-0.1)).argmin()
    max_idx = (np.abs(y-0.9)).argmin()
    loc = (x[max_idx] + x[min_idx])/2
    return loc

def plot_polys(var_scale,noise=True):
    noise_state = 'noise' if noise else ''
    for i in range(5):
        fig, axes = plt.subplots(5,2,figsize=(10,10))
        for ax in axes.ravel():
            plot_random_poly(ax,var_scale,noise)

        plt.tight_layout()
        filename = PROJECT_DIR + '/output/figures/Logit Exploration/logit_{}_{}_{}.png'.format(i,var_scale,noise_state)
        plt.savefig(filename)
        plt.close()

def plot_random_poly(ax,var_scale=1,noise=True):
    rng = np.random.default_rng()
    p = random_polynomial(rng)
    x = np.linspace(-2,2,200)
    f = p(x)
    signal_variance = np.var(f)
    noise_variance = signal_variance/var_scale
    if noise:
        epsilon = rng.normal(loc=0,scale=math.sqrt(noise_variance))
    else:
        epsilon = 0
    y = 1/(1+np.exp(-(f+epsilon)))
    min_idx = (np.abs(y-0.1)).argmin()
    max_idx = (np.abs(y-0.9)).argmin()
    loc = (x[max_idx] + x[min_idx])/2
    x = rng.normal(loc=loc,scale=1,size=200)
    x.sort()
    f = p(x)
    signal_variance = np.var(f)
    noise_variance = signal_variance/var_scale
    if noise:
        epsilon = rng.normal(loc=0,scale=math.sqrt(noise_variance))
    else:
        epsilon=0
    y = 1/(1+np.exp(-(f+epsilon)))
    ax.plot(x,y)

def poly_exploration():
    rng = np.random.default_rng(1)
    plt.close()
    p = random_polynomial(rng)
    x = np.linspace(-2,2,200)
    f = p(x)
    signal_variance = np.var(f)
    noise_variance = signal_variance/1000
    epsilon = rng.normal(loc=0,scale=math.sqrt(noise_variance))
    epsilon = 0
    y = 1/(1+np.exp(-(f+epsilon)))
    min_idx = (np.abs(y-0.1)).argmin()
    max_idx = (np.abs(y-0.9)).argmin()
    loc = (x[max_idx] + x[min_idx])/2
    x = rng.normal(loc=loc,scale=1,size=200)
    x.sort()
    f = p(x)
    signal_variance = np.var(f)
    noise_variance = signal_variance/1000
    epsilon = rng.normal(loc=0,scale=math.sqrt(noise_variance))
    epsilon=0
    y = 1/(1+np.exp(-(f+epsilon)))
    plt.plot(x,y)
    plt.show()

def random_polynomial(rng):
    coefs = rng.uniform(-1,1,8)
    coefs[::2] = 0
    p = Polynomial(coefs)
    return p

##### Optimization Functions #####
def min_CVaR_program(pred_y,train_y,params,include_premium=False):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    eps_p = params['epsilon_p']
    sizes = params['P']
    epsilon = params['epsilon']
    c_k = params['c_k']
    budget = params['B']

    n_samples = train_y.shape[0]
    n_zones = train_y.shape[1]
    S = np.tile(sizes,(n_samples,1))
    p = np.ones(n_samples)/n_samples
    A = cp.Variable((n_samples,n_zones))
    B = cp.Variable((n_samples,n_zones))
    t = cp.Variable((n_samples,n_zones))
    t_k = cp.Variable(n_samples)
    pi = cp.Variable((n_samples,n_zones))
    alpha = cp.Variable((n_samples,n_zones))
    omega = cp.Variable((n_samples,n_zones))
    gamma = cp.Variable((n_samples,n_zones))
    gamma_p = cp.Variable(n_samples)
    m = cp.Variable()
    K = cp.Variable()

    constraints = []

    # objective, m >= CVaR(l_z - I_z(theta_z))
    constraints.append(t[0,:] + (1/epsilon)*(p @ gamma) <= m*np.ones(n_zones))

    # CVaR constraints for each zone's loss, gamma^k_z >= l_z - min(a_z \hat{l_z}+b_z, K_z) - t
    if include_premium:
        constraints.append(gamma >= train_y + pi - omega -t)
    else:
        constraints.append(gamma >= train_y - omega -t)

    constraints.append(gamma >= 0)
    constraints.append(B <= 0)

    # Portfolio CVaR constraint: CVaR(\sum_z I_z(\theta_z)) <= K^P + \sum_z E[I_z(\theta_Z)]
    constraints.append(t_k + (1/eps_p)*(p @ gamma_p) <= K + (1/n_samples)*cp.sum(cp.multiply(S,omega)))
    constraints.append(gamma_p >= cp.sum(cp.multiply(S,alpha),axis=1)-t_k)
    constraints.append(gamma_p >= 0)
    constraints.append(omega <= cp.multiply(pred_y,A)+B)
    constraints.append(omega <= 1)
    constraints.append(alpha >= cp.multiply(pred_y,A)+B)
    constraints.append(alpha >= 0)

    # budget constraint
    constraints.append(budget >= (1/n_samples)*cp.sum(alpha) + c_k*K)

    # premium definition 
    if include_premium:
        constraints.append(pi[0,:] == (1/n_samples)*cp.sum(alpha,axis=0) + (1/np.sum(sizes)*c_k*K))

    for i in range(n_samples-1):
        constraints.append(A[i,:] == A[i+1,:])
        constraints.append(B[i,:] == B[i+1,:])
        constraints.append(t[i,:] == t[i+1,:])
        if include_premium:
            constraints.append(pi[i,:] == pi[i+1,:])

    objective = cp.Minimize(m)
    problem = cp.Problem(objective,constraints)
    if 'Projects' in os.getcwd():
        problem.solve(solver=cp.GUROBI)
    else:
        problem.solve(solver=cp.SCIPY, scipy_options={"method":"highs"})
    return (A.value[0,:], B.value[0,:])

##### Evaluation functions #####
def show_payout_functions(bdf,odf,a,b):
    n_zones = len(a)
    a = np.around(a,2)
    b = np.around(b,2)
    fig, axes = plt.subplots(n_zones,1,figsize=(10,6))
    for zone, ax in zip(range(n_zones), axes.ravel()):
        zbdf = bdf.loc[bdf.Zone == zone,:]
        zodf = odf.loc[odf.Zone == zone,:]
        ax.plot(zbdf['PredictedLosses'],zbdf['Premium'],'bs',label='baseline')
        ax.plot(zodf['PredictedLosses'],zodf['Premium'],'g^',label='opt')
        ax.plot(zbdf['PredictedLosses'],zbdf['Losses'],'ro',label='actual losses')
        ax.set_title('Zone: {}, a = {}, b = {}'.format(zone,a[zone],b[zone]))
        ax.legend()

    plt.show()

def plot_payout_functions(bdf,odf,a,b,axes,scenario=None):
    n_zones = len(a)
    a = np.around(a,2)
    b = np.around(b,2)
    for zone, ax in zip(range(n_zones), axes.ravel()):
        ax.plot(bdf['PredLossRate{}'.format(zone)],bdf['PayoutRate{}'.format(zone)],'bs',label='baseline')
        ax.plot(bdf['PredLossRate{}'.format(zone)],odf['PayoutRate{}'.format(zone)],'g^',label='opt')
        ax.plot(bdf['PredLossRate{}'.format(zone)],bdf['LossRate{}'.format(zone)],'ro',label='actual losses')
        ax.legend()
        if scenario is not None:
            ax.set_title('{}, a = {}, b = {}'.format(scenario,a[zone],b[zone]))
        else:
            ax.set_title('Zone: {}, a = {}, b = {}'.format(zone,a[zone],b[zone]))

def calculate_baseline_payouts(pred_y,strike_vals,Ps):
    n_zones = pred_y.shape[1]
    bdf = pd.DataFrame()
    for zone in np.arange(n_zones):
        pred_col = 'PredLossRate{}'.format(zone)
        payout_rate_col = 'PayoutRate{}'.format(zone)
        payout_col = 'Payout{}'.format(zone)

        bdf[pred_col] = pred_y[:,zone]
        bdf[payout_rate_col] = np.maximum(bdf[pred_col]-strike_vals[zone],0)
        bdf[payout_rate_col] = np.minimum(bdf[payout_rate_col],1)
        bdf[payout_col] = Ps[zone]*bdf[payout_rate_col]

    bdf['TotalPayout'] = bdf['Payout0'] + bdf['Payout1']
    return bdf

def calculate_opt_payouts(pred_y,a,b,Ps):
    n_zones = pred_y.shape[1]
    odf = pd.DataFrame()
    for zone in np.arange(n_zones):
        pred_col = 'PredLossRate{}'.format(zone)
        payout_rate_col = 'PayoutRate{}'.format(zone)
        payout_col = 'Payout{}'.format(zone)

        odf[pred_col] = pred_y[:,zone]
        odf[payout_rate_col] = np.minimum(a[zone]*odf[pred_col]+b[zone],1)
        odf[payout_rate_col] = np.maximum(0,odf[payout_rate_col])
        odf[payout_col] = Ps[zone]*odf[payout_rate_col]

    odf['TotalPayout'] = odf['Payout0'] + odf['Payout1']
    return odf

def make_eval_dfs(test_y,pred_y,strike_vals,a,b,Ps,premiums=None):
    n_zones = test_y.shape[1]
    bdf = calculate_baseline_payouts(pred_y,strike_vals,Ps)
    odf = calculate_opt_payouts(pred_y,a,b,Ps)

    for zone in np.arange(n_zones):
        payout_rate_col = 'PayoutRate{}'.format(zone)
        loss_rate_col = 'LossRate{}'.format(zone)
        net_loss_rate_col = 'NetLossRate{}'.format(zone)

        bdf[loss_rate_col] = test_y[:,zone]
        bdf[net_loss_rate_col] = bdf[loss_rate_col] - bdf[payout_rate_col]

        odf[loss_rate_col] = test_y[:,zone]
        odf[net_loss_rate_col] = odf[loss_rate_col] - odf[payout_rate_col]

        if premiums is not None:
            baseline_premiums = premiums['baseline']
            opt_premiums = premiums['opt']
            bdf[net_loss_rate_col] += baseline_premiums[zone]
            odf[net_loss_rate_col] += opt_premiums[zone]

    return bdf, odf

def calculate_premiums_naive(pred_y,strike_vals,a,b,params):    
    c_k = params['c_k']
    Ps = params['P']
    bdf = calculate_baseline_payouts(pred_y,strike_vals,Ps)
    odf = calculate_opt_payouts(pred_y,a,b,Ps)

    baseline_loss_cvar = CVaR(bdf,'TotalPayout','TotalPayout',0.01)
    baseline_average_payout = bdf['TotalPayout'].mean()
    baseline_required_capital = baseline_loss_cvar-baseline_average_payout
    baseline_premiums = bdf[['Payout0','Payout1']].mean() + 0.5*c_k*baseline_required_capital

    opt_loss_cvar = CVaR(odf,'TotalPayout','TotalPayout',0.01)
    opt_average_payout = odf['TotalPayout'].mean()
    opt_required_capital = opt_loss_cvar-opt_average_payout
    opt_premiums = odf[['Payout0','Payout1']].mean() + 0.5*c_k*opt_required_capital
    premiums = {'baseline':baseline_premiums,'opt':opt_premiums}
    return premiums

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
    return best_strike_percentiles, best_strike_vals

def determine_budget_params(pred_y,strike_vals,Ps,c_k=0.05):
    bdf = calculate_baseline_payouts(pred_y,strike_vals,Ps)

    bdf['TotalPayout'] = bdf['Payout0']+bdf['Payout1']
    payout_cvar = CVaR(bdf,'TotalPayout','TotalPayout',0.01)
    average_payout = bdf['TotalPayout'].mean()
    required_capital = payout_cvar-average_payout
    capital_costs = c_k*required_capital
    expected_total_costs = average_payout + capital_costs # this is the average total cost for one period
    return expected_total_costs, average_payout

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

def semi_variance(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    semi_variance = (df.loc[df[loss_col] >= q,outcome_col]-q)**2
    return semi_variance.sum()/(len(semi_variance)-1)

def get_summary_stats(df,cvar_eps=0.2):
    sdf = {}
    sdf['CVaR0'] = CVaR(df,'LossRate0','NetLossRate0',cvar_eps)
    sdf['CVaR1'] = CVaR(df,'LossRate1','NetLossRate1',cvar_eps)
    sdf['$|CVaR_2 - CVaR_1|$'] = np.abs(sdf['CVaR0']-sdf['CVaR1'])
    sdf['Max CVaR'] = np.maximum(sdf['CVaR0'],sdf['CVaR1'])

    sdf['VaR0'] = df['NetLossRate0'].quantile(1-cvar_eps)
    sdf['VaR1'] = df['NetLossRate1'].quantile(1-cvar_eps)
    sdf['$|VaR_2 - VaR_1|$'] = np.abs(sdf['VaR0']-sdf['VaR1'])
    sdf['Max VaR'] = np.maximum(sdf['VaR0'],sdf['VaR1'])

    sdf['SemiVar0'] = semi_variance(df,'LossRate0','NetLossRate0',cvar_eps)
    sdf['SemiVar1'] = semi_variance(df,'LossRate1','NetLossRate1',cvar_eps)
    sdf['Max SemiVar'] = np.maximum(sdf['SemiVar0'],sdf['SemiVar1'])

    sdf['Required Capital'] = CVaR(df,'TotalPayout','TotalPayout',0.01) - df['TotalPayout'].mean()    
    sdf['Payout CVaR'] = CVaR(df,'TotalPayout','TotalPayout',0.01)
    sdf['Average Cost'] = df['TotalPayout'].mean() + 0.15*sdf['Required Capital']
    sdf['Payout Cost'] = df['TotalPayout'].sum()
    return(sdf)

def get_summary_stats_bs(df):
    medians = {}
    confidence_interval = {}
    cols = ['Max CVaR','Max VaR','Max SemiVar','$|VaR_2 - VaR_1|$','Required Capital','Average Cost']
    for col in cols:
        medians[col] = np.around(df[col].median(),2)
        
        lower_bound = np.around(df[col].quantile(0.05),2)
        upper_bound = np.around(df[col].quantile(0.95),2)
        confidence_interval[col] = str([lower_bound,upper_bound])
        
    return(medians, confidence_interval)

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
    fig_filename = FIGURES_DIR + '/Logit Bootstrap/{}.png'.format(scenario_name)
    # need to get the index of the median in terms of outcome, get the corresponding a, b, and strike vals
    odf_median_VaR = odf['Max VaR'].quantile(interpolation='nearest')
    median_idx = odf.index[odf['Max VaR'] == odf_median_VaR][0]
    median_seed = odf.loc[median_idx,'seed']
    a = np.zeros(2)
    b = np.zeros(2)

    a[0] = odf.loc[median_idx,'a_1']
    a[1] = odf.loc[median_idx,'a_2']
    b[0] = odf.loc[median_idx,'b_1']
    b[1] = odf.loc[median_idx,'b_1']

    opt_premiums = odf.loc[median_idx,['p1','p2']].to_numpy()
    baseline_premiums = bdf.loc[median_idx,['p1','p2']].to_numpy()
    premiums = {'baseline':baseline_premiums,'opt':opt_premiums}

    strike_vals = np.zeros(2)
    strike_vals[0] = bdf.loc[median_idx,'s_1']
    strike_vals[1] = bdf.loc[median_idx,'s_2']

    nonlinear = 'nonlinear' in scenario_name
    y, X = make_multi_zone_logit_data(sigma,n=500,nonlinear=nonlinear,seed=median_seed)
    train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.33,random_state=median_seed)
    pred_model = LinearRegression().fit(train_x,train_y)
    pred_test_y = pred_model.predict(test_x)
    ebdf, eodf = make_eval_dfs(test_y,pred_test_y,strike_vals,a,b,params['P'],premiums)

    fig, axes = plt.subplots(1,2,figsize=(10,6))
    plot_payout_functions(ebdf,eodf,a,b,axes)
    fig.savefig(fig_filename)
    plt.close()

def save_bootstrap_results(bdf,odf,scenario_name):
    table_filename = TABLES_DIR + '/Logit Bootstrap/{}.tex'.format(scenario_name)

    sdf = bootstrap_comparison_df(bdf,odf)
    sdf.to_latex(table_filename,float_format="{:0.2f}".format,escape=False,index=False)

##### Exploration functions #####
def bootstrap_scenario_exploration(regime,P,include_premium=False,n=1000):
    premium_status = 'premium' if include_premium else 'no_premium'

    # No correlation case
    scenario_name = 'no_corr_{}_{}'.format(regime,premium_status)
    sigma = np.array(np.array([[1,0],[0,1]]))
    cvar_eps=0.2
    params = {'epsilon':cvar_eps,'epsilon_p':0.01,'P':np.array([P,P]),'rho':np.array([0.5,0.5]),'c_k':0.15}

    baseline_results = []
    opt_results = []
    for i in range(n):
        bdict, odict = run_bootstrap_scenario(sigma,params,regime,seed=i)
        baseline_results.append(bdict)
        opt_results.append(odict)

    odf = pd.DataFrame(opt_results)
    bdf = pd.DataFrame(baseline_results)
    # print(bootstrap_comparison_df(bdf,odf))
    plot_bootstrap_results(sigma,params,bdf,odf,scenario_name)
    save_bootstrap_results(bdf,odf,scenario_name)

    # Positive Correlation
    scenario_name = 'pos_corr_{}_{}'.format(regime,premium_status)
    sigma = np.array(np.array([[1,0.9],[0.9,1]]))
    baseline_results = []
    opt_results = []
    for i in range(n):
        bdict, odict = run_bootstrap_scenario(sigma,params,regime,seed=i)
        baseline_results.append(bdict)
        opt_results.append(odict)

    odf = pd.DataFrame(opt_results)
    bdf = pd.DataFrame(baseline_results)
    # print(bootstrap_comparison_df(bdf,odf))
    plot_bootstrap_results(sigma,params,bdf,odf,scenario_name)
    save_bootstrap_results(bdf,odf,scenario_name)

    # Negative Correlation
    scenario_name = 'neg_corr_{}_{}'.format(regime,premium_status)
    sigma = np.array(np.array([[1,-0.9],[-0.9,1]]))
    baseline_results = []
    opt_results = []
    for i in range(n):
        bdict, odict = run_bootstrap_scenario(sigma,params,regime,seed=i)
        baseline_results.append(bdict)
        opt_results.append(odict)

    odf = pd.DataFrame(opt_results)
    bdf = pd.DataFrame(baseline_results)
    # print(bootstrap_comparison_df(bdf,odf))
    plot_bootstrap_results(sigma,params,bdf,odf,scenario_name)
    save_bootstrap_results(bdf,odf,scenario_name)

def run_bootstrap_scenario(sigma,params,scenario_name,include_premium=False,seed=1):
    nonlinear = 'nonlinear' in scenario_name
    y, X = make_multi_zone_logit_data(sigma,n=500,nonlinear=nonlinear,seed=seed)
    train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.33,random_state=seed)
    test_x, eval_x, test_y, eval_y =train_test_split(test_x,test_y,test_size=0.33,random_state=seed)

    pred_model = LinearRegression().fit(train_x,train_y)
    pred_train_y = pred_model.predict(train_x)
    pred_test_y = pred_model.predict(test_x)
    strike_ps, strike_vals = determine_strike_values(train_y,eval_y,eval_x,pred_model)
    params['B'], params['premium income'] = determine_budget_params(pred_train_y,strike_vals,params['P'],params['c_k'])

    a,b = np.around(min_CVaR_program(pred_train_y,train_y,params),2)
    premiums = calculate_premiums_naive(pred_train_y,strike_vals,a,b,params)

    if include_premium:
        bdf, odf = make_eval_dfs(test_y,pred_test_y,strike_vals,a,b,params['P'],premiums)
    else:
        bdf, odf = make_eval_dfs(test_y,pred_test_y,strike_vals,a,b,params['P'])

    bdict = get_summary_stats(bdf,params['epsilon'])
    odict = get_summary_stats(odf,params['epsilon'])
    
    bdict['s_1'], bdict['s_2'] = strike_vals
    odict['a_1'], odict['a_2'] = a
    odict['b_1'], odict['b_2'] = b
    odict['p1'], odict['p2'] = premiums['opt']
    bdict['p1'], bdict['p2'] = premiums['baseline']
    odict['seed'] = seed 
    bdict['seed'] = seed
    return(bdict,odict)

def debugging():
    regime = 'nonlinear'
    P = 1
    include_premium = True 
    n=10
    premium_status = 'premium' if include_premium else 'no_premium'

    # No correlation case
    scenario_name = 'no_corr_{}_{}'.format(regime,premium_status)
    sigma = np.array(np.array([[1,0],[0,1]]))
    cvar_eps=0.2
    params = {'epsilon':cvar_eps,'epsilon_p':0.01,'P':np.array([P,P]),'rho':np.array([0.5,0.5]),'c_k':0.15}

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

    # Negative Correlation
    scenario_name = 'neg_corr_{}_{}'.format(regime,premium_status)
    sigma = np.array(np.array([[1,-0.9],[-0.9,1]]))
    baseline_results = []
    opt_results = []
    for i in range(n):
        bdict, odict = run_bootstrap_scenario(sigma,params,regime)
        baseline_results.append(bdict)
        opt_results.append(odict)

    odf = pd.DataFrame(opt_results)
    bdf = pd.DataFrame(baseline_results)
    print(bootstrap_comparison_df(bdf,odf))

start = time.time()
bootstrap_scenario_exploration('linear',1,True,1000)
bootstrap_scenario_exploration('nonlinear',1,True,1000)
# bootstrap_scenario_exploration('linear',8,False,100)
# bootstrap_scenario_exploration('nonlinear',45,False,100)
end = time.time()
print(end-start)



