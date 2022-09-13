import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import cvxpy as cp
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

FIGURES_DIR = PROJECT_DIR + '/output/figures'
TABLES_DIR = PROJECT_DIR + '/output/tables'

# TODO: 1. figure out how to include positive and negative losses as well
# 2. add function to create a covariance matrix based on the desired correlation
# 3. change code to train a separate regression model for each zone.

##### Data Creation #####
def make_multi_zone_data(beta,mu,sigma,n=100):
    dim = len(mu)
    X = np.random.multivariate_normal(mean=mu,cov=sigma,size=n)
    epsilon = np.random.multivariate_normal(mean=np.zeros(dim),cov=np.eye(dim),size=n)
    y = np.dot(X**2,beta) + epsilon
    return y, X

##### Optimization Functions #####
def min_CVaR_program(pred_y,train_y,params):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    max_pi = params['max_premium']
    min_pi = params['min_premium']
    eps_p = params['epsilon_p']
    Ks = params['K']
    epsilon = params['epsilon']
    rhos = params['rho']
    c_k = params['c_k']
    budget = params['B']
    premium_income_sq = params['premium income']

    n_samples = train_y.shape[0]
    n_zones = train_y.shape[1]
    Ks = np.tile(Ks,(n_samples,1))
    p = np.ones(n_samples)/n_samples
    A = cp.Variable((n_samples,n_zones))
    B = cp.Variable((n_samples,n_zones))
    t = cp.Variable((n_samples,n_zones))
    t_p = cp.Variable(n_samples)
    alpha = cp.Variable((n_samples,n_zones))
    gamma = cp.Variable((n_samples,n_zones))
    gamma_p = cp.Variable(n_samples)
    m = cp.Variable()
    K_p = cp.Variable()

    constraints = []

    # objective, m >= CVaR(l_z - I_z(theta_z))
    constraints.append(t[0,:] + (1/epsilon)*(p @ gamma) <= m*np.ones(n_zones))

    # CVaR constraints for each zone's loss, gamma^k_z >= l_z - min(a_z \hat{l_z}+b_z, K_z) - t
    constraints.append(gamma >= train_y - cp.multiply(pred_y,A) -B -t)
    constraints.append(gamma >= train_y - Ks -t)
    constraints.append(gamma >= 0)

    # Portfolio CVaR constraint: CVaR(\sum_z I_z(\theta_z)) <= K^P + \sum_z E[I_z(\theta_Z)]
    constraints.append(t_p + (1/eps_p)*(p @ gamma_p) <= K_p + premium_income_sq)
    constraints.append(gamma_p >= cp.sum(alpha,axis=1)-t_p)
    constraints.append(gamma_p >= 0)
    constraints.append(alpha >= cp.multiply(pred_y,A)+B)
    constraints.append(alpha >= 0)

    # budget constraint
    constraints.append(budget >= cp.sum(alpha) + c_k*K_p)

    for i in range(n_samples-1):
        constraints.append(A[i,:] == A[i+1,:])
        constraints.append(B[i,:] == B[i+1,:])
        constraints.append(t[i,:] == t[i+1,:])

    objective = cp.Minimize(m)
    problem = cp.Problem(objective,constraints)
    problem.solve(solver=cp.GLPK)
    print('Req capital: {}'.format(K_p.value))
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
        zbdf = bdf.loc[bdf.Zone == zone,:]
        zodf = odf.loc[odf.Zone == zone,:]
        ax.plot(zbdf['PredictedLosses'],zbdf['Premium'],'bs',label='baseline')
        ax.plot(zodf['PredictedLosses'],zodf['Premium'],'g^',label='opt')
        ax.plot(zbdf['PredictedLosses'],zbdf['Losses'],'ro',label='actual losses')
        ax.legend()
        if scenario is not None:
            ax.set_title('{}, a = {}, b = {}'.format(scenario,a[zone],b[zone]))
        else:
            ax.set_title('Zone: {}, a = {}, b = {}'.format(zone,a[zone],b[zone]))

def plot_payout_functions2(bdf,odf,a,b,axes,scenario=None):
    n_zones = len(a)
    a = np.around(a,2)
    b = np.around(b,2)
    for zone, ax in zip(range(n_zones), axes.ravel()):
        ax.plot(bdf['PredictedLosses{}'.format(zone)],bdf['Payout{}'.format(zone)],'bs',label='baseline')
        ax.plot(odf['PredictedLosses{}'.format(zone)],odf['Payout{}'.format(zone)],'g^',label='opt')
        ax.plot(bdf['PredictedLosses{}'.format(zone)],bdf['Losses{}'.format(zone)],'ro',label='actual losses')
        ax.legend()
        if scenario is not None:
            ax.set_title('{}, a = {}, b = {}'.format(scenario,a[zone],b[zone]))
        else:
            ax.set_title('Zone: {}, a = {}, b = {}'.format(zone,a[zone],b[zone]))

def make_eval_dfs(test_x,test_y,pred_model,strike_vals,a,b,Ks):
    col_names = ['P(L > {})'.format(i) for i in [70,80,90,95]]
    n_zones = test_y.shape[1]
    
    bdfs = []
    odfs = []
    pred_y = pred_model.predict(test_x)
    for zone in np.arange(n_zones):
        bdf = pd.DataFrame()
        bdf['Losses'] = test_y[:,zone]
        bdf['PredictedLosses'] = pred_y[:,zone]
        bdf['Payout'] = np.maximum(bdf['PredictedLosses']-strike_vals[zone],0)
        bdf['NetLoss'] = bdf['Losses'] - bdf['Payout']
        
        odf = pd.DataFrame()
        odf['Losses'] = test_y[:,zone]
        odf['PredictedLosses'] = pred_y[:,zone]
        odf['Payout'] = np.minimum(a[zone]*odf['PredictedLosses']+b[zone],Ks[zone])
        odf['Payout'] = np.maximum(0,odf['Payout'])
        odf['NetLoss'] = odf['Losses']-odf['Payout']

        loss_quantiles = np.quantile(test_y[:,zone],[0.7,0.8,0.9,0.95])
        for col_name, loss_quantile in zip(col_names,loss_quantiles):
            bdf[col_name] = bdf['NetLoss'] > loss_quantile
            odf[col_name] = odf['NetLoss'] > loss_quantile

        odf = odf.rename(columns={'Payout':'Premium'})
        bdf = bdf.rename(columns={'Payout':'Premium'})
        bdf['Zone'] = zone
        odf['Zone'] = zone
        bdfs.append(bdf)
        odfs.append(odf)
    bdf = pd.concat(bdfs,ignore_index=True)
    odf = pd.concat(odfs,ignore_index=True)
    return bdf, odf

def make_eval_dfs2(test_x,test_y,pred_model,strike_vals,a,b,Ks):
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
        bdf[payout_col] = np.minimum(bdf[payout_col],Ks[zone])
        bdf[loss_col] = test_y[:,zone]
        bdf[net_loss_col] = bdf[loss_col]-bdf[payout_col]

        odf[pred_col] = pred_y[:,zone]
        odf[payout_col] = np.minimum(a[zone]*odf[pred_col]+b[zone],Ks[zone])
        odf[payout_col] = np.maximum(0,odf[payout_col])
        odf[loss_col] = test_y[:,zone]
        odf[net_loss_col] = odf[loss_col]-odf[payout_col]

    odf['TotalPayout'] = odf['Payout0'] + odf['Payout1']
    odf['TotalLoss'] = odf['Losses0'] + odf['Losses1']
    odf['TotalNetLoss'] = odf['NetLoss0'] + odf['NetLoss1']

    bdf['TotalPayout'] = bdf['Payout0'] + bdf['Payout1']
    bdf['TotalLoss'] = bdf['Losses0'] + bdf['Losses1']
    bdf['TotalNetLoss'] = bdf['NetLoss0'] + bdf['NetLoss1']

    return bdf, odf

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

def determine_budget_params(pred_y,strike_vals,Ks,c_k=0.05):
    bdf = pd.DataFrame()
    n_zones = pred_y.shape[1]
    for zone in np.arange(n_zones):
        pred_col = 'PredictedLosses{}'.format(zone)
        payout_col = 'Payout{}'.format(zone)
        bdf[pred_col] = pred_y[:,zone]
        bdf[payout_col] = np.maximum(bdf[pred_col]-strike_vals[zone],0)
        bdf[payout_col] = np.minimum(bdf[payout_col],Ks[zone])

    bdf['Total Payouts'] = bdf['Payout0']+bdf['Payout1']
    loss_quantile = np.quantile(bdf['Total Payouts'],0.99)
    average_premium = bdf['Total Payouts'].mean()
    payment_costs = bdf['Total Payouts'].sum()
    required_capital = loss_quantile-average_premium
    print('Required Capital: {}'.format(required_capital))
    capital_costs = c_k*required_capital
    total_costs = payment_costs + capital_costs
    return total_costs, average_premium

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

def get_summary_stats(df,cvar_eps=0.2):
    sdf = {}
    sdf['CVaR0'] = CVaR(df,'Losses0','NetLoss0',cvar_eps)
    sdf['CVaR1'] = CVaR(df,'Losses1','NetLoss1',cvar_eps)
    sdf['CVaR Total'] = CVaR(df,'TotalLoss','TotalNetLoss',cvar_eps)
    sdf['CVaR Diff'] = np.abs(sdf['CVaR0']-sdf['CVaR1'])

    sdf['NetLoss0'] = df['NetLoss0'].mean()
    sdf['NetLoss1'] = df['NetLoss1'].mean()
    sdf['NetTotal'] = df['TotalNetLoss'].mean()
    sdf['Loss Diff'] = np.abs(sdf['NetLoss0']-sdf['NetLoss1'])

    col_names = ['P(L > {})'.format(i) for i in [50,60,70,80]]
    loss_quantiles = np.quantile(df['Losses0'],[0.5,0.6,0.7,0.8])
    for col_name, loss_quantile in zip(col_names,loss_quantiles):
        sdf[col_name] = ((df['NetLoss0'] > loss_quantile) | (df['NetLoss1'] > loss_quantile)).mean()

    sdf['Required Capital'] = CVaR(df,'TotalPayout','TotalPayout',0.01) - df['TotalPayout'].mean()    
    sdf['Payout CVaR'] = CVaR(df,'TotalPayout','TotalPayout',0.01)
    sdf['TotalCost'] = df['TotalPayout'].sum() + 0.05*sdf['Required Capital']
    return(sdf)

def comparison_df(bdf,odf,cvar_eps=0.2):
    bdict = get_summary_stats(bdf,cvar_eps)
    odict = get_summary_stats(odf,cvar_eps)
    sdf = pd.DataFrame([bdict,odict],index=['Baseline','Opt'])
    return sdf

##### Exploration functions #####
def run_scenario(beta,mu,sigma,params,axes,scenario_name):
    train_y, train_x = make_multi_zone_data(beta,mu,sigma,n=1000)
    eval_y, eval_x = make_multi_zone_data(beta,mu,sigma,n=100)
    test_y, test_x = make_multi_zone_data(beta,mu,sigma,n=400)

    pred_model = LinearRegression().fit(train_x,train_y)
    pred_y = pred_model.predict(train_x)
    strike_ps, strike_vals = determine_strike_values(train_y,eval_y,eval_x,pred_model)
    print('strike vals: {}, {}'.format(strike_vals[0],strike_vals[1]))
    params['B'], params['premium income'] = determine_budget_params(pred_y,strike_vals,params['K'],params['c_k'])

    a,b = np.around(min_CVaR_program(pred_y,train_y,params),2)
    print('a: {}, b: {}'.format(a,b))
    bdf, odf = make_eval_dfs2(test_x,test_y,pred_model,strike_vals,a,b,params['K'])
    # plot_payout_functions2(bdf,odf,a,b,axes,scenario_name)
    # return(a,b)
    return(bdf,odf)

def scenario_exploration():
    fig, axes = plt.subplots(4,2,figsize=(10,10))
    default_params = {'max_premium':0.6,'min_premium':0.1,'epsilon':0.2,'epsilon_p':0.05,
        'K':np.array([8,8]),'rho':np.array([0.5,0.5]),'c_k':0.05}
    beta = np.array([[1.5,0],[0,1.5]])
    mu = np.array([5,5])

    # No correlation case
    fig, axes = plt.subplots(1,2,figsize=(10,6))
    sigma = np.array(np.array([[2,0],[0,2]]))
    cvar_eps=0.2
    K = 8
    params = {'max_premium':0.6,'min_premium':0.05,'epsilon':cvar_eps,'epsilon_p':0.01,
            'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.05}
    filepath = FIGURES_DIR + '/Exploration/no_correlation.png'
    bdf, odf = run_scenario(beta,mu,sigma,params,axes, 'No corr')
    comparison_df(bdf,odf,cvar_eps)
    plt.show()

    # Positive Correlation
    fig, axes = plt.subplots(1,2,figsize=(10,6))
    sigma = np.array(np.array([[2,1.9],[1.9,2]]))
    cvar_eps=0.2
    K = 45
    params = {'max_premium':0.6,'min_premium':0.05,'epsilon':cvar_eps,'epsilon_p':0.01,
            'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.05}
    filepath = FIGURES_DIR + '/Exploration/pos_correlation.png'
    bdf, odf = run_scenario(beta,mu,sigma,params,axes,'Pos Corr')
    comparison_df(bdf,odf,cvar_eps)
    plt.show()

    # Negative Correlation
    fig, axes = plt.subplots(1,2,figsize=(10,6))
    sigma = np.array(np.array([[2,-1.9],[-1.9,2]]))
    cvar_eps=0.2
    params = {'max_premium':0.6,'min_premium':0.05,'epsilon':cvar_eps,'epsilon_p':0.01,
        'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.05}
    filepath = FIGURES_DIR + '/Exploration/neg_correlation.png'
    bdf, odf = run_scenario(beta,mu,sigma,params,axes,'Neg corr')
    sdf = comparison_df(bdf,odf)
    plt.show()

    # Disparate variance
    sigma = np.array(np.array([[2,0],[0,4]]))
    cvar_eps=0.2
    K = 8
    params = {'max_premium':0.6,'min_premium':0.05,'epsilon':cvar_eps,'epsilon_p':0.01,
            'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.05}
    filepath = FIGURES_DIR + '/Exploration/disp_variance.png'
    bdf, odf = run_scenario(beta,mu,sigma,params,axes,'Disp var')
    comparison_df(bdf,odf,cvar_eps)

    plt.tight_layout()
    filename = FIGURES_DIR + '/Exploration/scenario_exploration.png'
    plt.savefig(filename)

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    x = np.linspace(0,16,50)
    for zone, ax in zip(range(2),axes.ravel()):
        ax.plot(x,ab[zone]*x+bb[zone],'g',label='no corr')
        ax.plot(x,apc[zone]*x+bpc[zone],'b',label='pos corr')
        ax.plot(x,anc[zone]*x+bnc[zone],'p',label='neg corr')
        ax.legend()
        ax.set_title('Zone: {}'.format(zone))

    plt.tight_layout()
    filename = FIGURES_DIR + '/Exploration/scenario_payouts_by_zone.png'
    plt.savefig(filename)




