import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import time

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

FIGURES_DIR = PROJECT_DIR + '/output/figures'
TABLES_DIR = PROJECT_DIR + '/output/tables'

# TODO: 1. figure out how to include positive and negative losses as well
# 2. add function to create a covariance matrix based on the desired correlation
# 3. change code to train a separate regression model for each zone.
# 4. modify min_CVaR_program to include rhos, separate budget constrant into two, one for each zone,
# make each zone's budget \sum I_z + \rho c_k B, think about how to set budget for each zone?

##### Data Creation #####
def make_multi_zone_data(beta,mu,sigma,n=100,square=False):
    dim = len(mu)
    X = np.random.multivariate_normal(mean=mu,cov=sigma,size=n)
    epsilon = np.random.multivariate_normal(mean=np.zeros(dim),cov=np.eye(dim),size=n)
    if square:
        y = np.dot(X**2,beta) + epsilon
    else:
        y = np.dot(X,beta) + epsilon
    return y, X

##### Optimization Functions #####
def min_CVaR_program(pred_y,train_y,params):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    eps_p = params['epsilon_p']
    Ks = params['K']
    epsilon = params['epsilon']
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
    constraints.append(budget >= (1/n_samples)*cp.sum(alpha) + c_k*K_p)

    for i in range(n_samples-1):
        constraints.append(A[i,:] == A[i+1,:])
        constraints.append(B[i,:] == B[i+1,:])
        constraints.append(t[i,:] == t[i+1,:])

    objective = cp.Minimize(m)
    problem = cp.Problem(objective,constraints)
    problem.solve(solver=cp.SCIPY, scipy_options={"method":"highs"})
    # print('Req capital: {}'.format(K_p.value))
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
        ax.plot(bdf['PredictedLosses{}'.format(zone)],bdf['Payout{}'.format(zone)],'bs',label='baseline')
        ax.plot(bdf['PredictedLosses{}'.format(zone)],odf['Payout{}'.format(zone)],'g^',label='opt')
        ax.plot(bdf['PredictedLosses{}'.format(zone)],bdf['Losses{}'.format(zone)],'ro',label='actual losses')
        ax.legend()
        if scenario is not None:
            ax.set_title('{}, a = {}, b = {}'.format(scenario,a[zone],b[zone]))
        else:
            ax.set_title('Zone: {}, a = {}, b = {}'.format(zone,a[zone],b[zone]))

def make_payout_dfs(test_x,test_y,pred_model,strike_vals,a,b,params):
    n_zones = test_y.shape[1]
    
    # offset = params['offset']
    Ks = params['K']
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
    average_payout = bdf['Total Payouts'].mean()
    required_capital = loss_quantile-average_payout
    # print('Required Capital: {}'.format(required_capital))
    capital_costs = c_k*required_capital
    average_total_costs = average_payout + capital_costs
    return average_total_costs, average_payout

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

def get_summary_stats(df,cvar_eps=0.2):
    sdf = {}
    # sdf['CVaR0'] = CVaR(df,'Losses0','NetLoss0',cvar_eps)
    # sdf['CVaR1'] = CVaR(df,'Losses1','NetLoss1',cvar_eps)
    # sdf['Total CVaR'] = CVaR(df,'TotalLoss','TotalNetLoss',cvar_eps)
    # sdf['$|CVaR_2 - CVaR_1|$'] = np.abs(sdf['CVaR0']-sdf['CVaR1'])
    # sdf['Max CVaR'] = np.maximum(sdf['CVaR0'],sdf['CVaR1'])

    sdf['VaR0'] = df['NetLoss0'].quantile(1-cvar_eps)
    sdf['VaR1'] = df['NetLoss1'].quantile(1-cvar_eps)
    # sdf['Total CVaR'] = CVaR(df,'TotalLoss','TotalNetLoss',cvar_eps)
    sdf['$|VaR_2 - VaR_1|$'] = np.abs(sdf['VaR0']-sdf['VaR1'])
    sdf['Max VaR'] = np.maximum(sdf['VaR0'],sdf['VaR1'])

    sdf['NetLoss0'] = df['NetLoss0'].mean()
    sdf['NetLoss1'] = df['NetLoss1'].mean()
    # sdf['NetTotal'] = df['TotalNetLoss'].mean()
    sdf['$|L_2 - L_1|$'] = np.abs(sdf['NetLoss0']-sdf['NetLoss1'])

    col_names = ['$P(L > Q({}))$'.format(i) for i in [.60]]
    loss_quantiles = np.quantile(df['Losses0'],[0.6])
    for col_name, loss_quantile in zip(col_names,loss_quantiles):
        sdf[col_name] = ((df['NetLoss0'] > loss_quantile) | (df['NetLoss1'] > loss_quantile)).mean()

    sdf['Required Capital'] = CVaR(df,'TotalPayout','TotalPayout',0.01) - df['TotalPayout'].mean()    
    sdf['Payout CVaR'] = CVaR(df,'TotalPayout','TotalPayout',0.01)
    sdf['Average Cost'] = df['TotalPayout'].mean() + 0.15*sdf['Required Capital']
    sdf['Payout Cost'] = df['TotalPayout'].sum()
    return(sdf)

def comparison_df(bdf,odf,cvar_eps=0.2):
    bdict = get_summary_stats(bdf,cvar_eps)
    odict = get_summary_stats(odf,cvar_eps)
    sdf = pd.DataFrame([bdict,odict],index=['Baseline','Opt'])
    sdf.drop(columns=['NetLoss0','NetLoss1','Payout CVaR'],inplace=True)
    return sdf

##### Exploration functions #####
def run_scenario(beta,mu,sigma,params,axes,scenario_name):
    square = 'nonlinear' in scenario_name
    train_y, train_x = make_multi_zone_data(beta,mu,sigma,n=300,square=square)
    eval_y, eval_x = make_multi_zone_data(beta,mu,sigma,n=50,square=square)
    test_y, test_x = make_multi_zone_data(beta,mu,sigma,n=100,square=square)

    pred_model = LinearRegression().fit(train_x,train_y)
    pred_y = pred_model.predict(train_x)
    strike_ps, strike_vals = determine_strike_values(train_y,eval_y,eval_x,pred_model)
    # print('strike vals: {}, {}'.format(strike_vals[0],strike_vals[1]))
    params['B'], params['premium income'] = determine_budget_params(pred_y,strike_vals,params['K'],params['c_k'])

    a,b = np.around(min_CVaR_program(pred_y,train_y,params),2)
    print('a: {}, b: {}'.format(a,b))
    bdf, odf = make_payout_dfs(test_x,test_y,pred_model,strike_vals,a,b,params)
    plot_payout_functions(bdf,odf,a,b,axes,scenario_name)
    # return(a,b)
    return(bdf,odf)

def scenario_exploration(regime,K):
    beta = np.array([[1.5,0],[0,1.5]])
    mu = np.array([5,5])

    # No correlation case
    fig, axes = plt.subplots(1,2,figsize=(10,6))

    sigma = np.array(np.array([[2,0],[0,2]]))
    cvar_eps=0.05
    params = {'max_premium':0.6,'min_premium':0.05,'epsilon':cvar_eps,'epsilon_p':0.01,
            'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.05,'offset':20}

    fig_filename = FIGURES_DIR + '/Exploration/no_correlation_{}.png'.format(regime)
    table_filename = TABLES_DIR + '/Exploration/no_correlation_{}.tex'.format(regime)

    bdf, odf = run_scenario(beta,mu,sigma,params,axes, 'No corr, {}'.format(regime))
    sdf = comparison_df(bdf,odf,cvar_eps)
    sdf.to_latex(table_filename,float_format='%.2f',escape=False)
    fig.savefig(fig_filename)
    plt.close()

    # Positive Correlation
    fig, axes = plt.subplots(1,2,figsize=(10,6))
    sigma = np.array(np.array([[2,1.6],[1.6,2]]))
    cvar_eps=0.05
    params = {'max_premium':0.6,'min_premium':0.05,'epsilon':cvar_eps,'epsilon_p':0.01,
            'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.05,'offset':20}

    fig_filename = FIGURES_DIR + '/Exploration/pos_correlation_{}.png'.format(regime)
    table_filename = TABLES_DIR + '/Exploration/pos_correlation_{}.tex'.format(regime)

    bdf, odf = run_scenario(beta,mu,sigma,params,axes, 'Pos corr, {}'.format(regime))
    sdf = comparison_df(bdf,odf,cvar_eps)
    sdf.to_latex(table_filename,float_format='%.2f',escape=False)
    fig.savefig(fig_filename)
    plt.close()

    # Negative Correlation
    fig, axes = plt.subplots(1,2,figsize=(10,6))
    sigma = np.array(np.array([[2,-1.6],[-1.6,2]]))
    cvar_eps=0.05
    params = {'max_premium':0.6,'min_premium':0.05,'epsilon':cvar_eps,'epsilon_p':0.01,
        'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.05}

    fig_filename = FIGURES_DIR + '/Exploration/neg_correlation_{}.png'.format(regime)
    table_filename = TABLES_DIR + '/Exploration/neg_correlation_{}.tex'.format(regime)

    bdf, odf = run_scenario(beta,mu,sigma,params,axes, 'Neg corr, {}'.format(regime))
    sdf = comparison_df(bdf,odf,cvar_eps)
    sdf.to_latex(table_filename,float_format='%.2f',escape=False)
    fig.savefig(fig_filename)
    plt.close()

    # Disparate variance
    fig, axes = plt.subplots(1,2,figsize=(10,6))
    sigma = np.array(np.array([[2,0],[0,4]]))
    cvar_eps=0.1
    params = {'max_premium':0.6,'min_premium':0.05,'epsilon':cvar_eps,'epsilon_p':0.01,
            'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.05}

    fig_filename = FIGURES_DIR + '/Exploration/disp_var_{}.png'.format(regime)
    table_filename = TABLES_DIR + '/Exploration/disp_var_{}.tex'.format(regime)

    bdf, odf = run_scenario(beta,mu,sigma,params,axes, 'Disp Var, {}'.format(regime))
    sdf = comparison_df(bdf,odf,cvar_eps)
    sdf.to_latex(table_filename,float_format='%.2f',escape=False)
    fig.savefig(fig_filename)
    plt.close()

def epsilon_exploration(regime,K):
    beta = np.array([[1.5,0],[0,1.5]])
    mu = np.array([5,5])
    sigma = np.array(np.array([[2,0],[0,2]]))
    eps_vals = np.linspace(0.3,0.01,30)
    fig, axes = plt.subplots(2,2,figsize=(10,6))
    sdf_list = []
    for eps in eps_vals:
        cvar_eps=eps
        params = {'max_premium':0.6,'min_premium':0.05,'epsilon':cvar_eps,'epsilon_p':0.01,
                'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.05}

        bdf, odf = run_scenario(beta,mu,sigma,params,axes, 'No corr, {}'.format(regime))
        sdf = comparison_df(bdf,odf,cvar_eps)
        sdf_list.append(sdf)

    sdf = pd.concat(sdf_list)

    metrics = ['Total CVaR','CVaR Diff','Max CVaR', '$P(L > Q(0.6))$']
    fig, axes = plt.subplots(2,2,figsize=(10,6))
    for metric, ax in zip(metrics,axes.ravel()):
        ax.plot(sdf.loc[sdf['index'] == 'Opt','Eps'],sdf.loc[sdf['index'] =='Opt',metric],'g^',label='opt')
        ax.plot(sdf.loc[sdf['index'] == 'Baseline','Eps'],sdf.loc[sdf['index']=='Baseline',metric],'bs',label='baseline')
        ax.legend()
        ax.set_title(metric)
        ax.invert_xaxis()
    plt.tight_layout()
    plt.show()
    new_eps = []
    for eps in eps_vals:
        new_eps.append(np.around(eps,2))
        new_eps.append(np.around(eps,2))




