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
# 2. figure out how to set w_bar when w is supposed to be loss

##### Data Creation #####
def make_multi_zone_data(beta,mu,sigma,n=100):
    dim = len(mu)
    X = np.random.multivariate_normal(mean=mu,cov=sigma,size=n)
    epsilon = np.random.multivariate_normal(mean=np.zeros(dim),cov=np.eye(dim),size=n)
    y = np.dot(X,beta) + epsilon
    return y, X

##### Optimization Functions #####
def min_CVaR_program(pred_y,train_y,params):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    max_pi = params['max_premium']
    min_pi = params['min_premium']
    eps_p = params['epsilon_p']
    Ks = params['K']
    epsilon = params['epsilon']
    betas = params['beta']
    c_k = params['c_k']

    n_samples = train_y.shape[0]
    n_zones = train_y.shape[1]
    Ks = np.tile(Ks,(n_samples,1))
    p = np.ones(n_samples)/n_samples
    A = cp.Variable((n_samples,n_zones))
    B = cp.Variable((n_samples,n_zones))
    t = cp.Variable((n_samples,n_zones))
    t_p = cp.Variable(n_samples)
    gamma = cp.Variable((n_samples,n_zones))
    gamma_p = cp.Variable(n_samples)
    m = cp.Variable()
    K_p = cp.Variable()

    constraints = []
    constraints.append(t[0,:] + (1/epsilon)*(p @ gamma) <= m*np.ones(n_zones))
    constraints.append(gamma >= train_y - cp.multiply(pred_y,A) -B -t)
    constraints.append(gamma >= train_y - Ks -t)
    constraints.append(gamma >= 0)
    constraints.append(t_p + (1/eps_p)*(p @ gamma_p) <= K_p + n_zones*min_pi)
    constraints.append(gamma_p >= cp.sum(cp.multiply(pred_y,A) +B,axis=1) -t_p)
    constraints.append(gamma_p >= 0)
    constraints.append(Ks[0,:]*max_pi >= cp.multiply(A[0,:]*(1/n_samples),cp.sum(pred_y,axis=0))+B[0,:] + c_k*K_p*betas)
    constraints.append(cp.multiply(pred_y,A)+B >= 0)
    constraints.append(A >= 0)

    for i in range(n_samples-1):
        constraints.append(A[i,:] == A[i+1,:])
        constraints.append(B[i,:] == B[i+1,:])
        constraints.append(t[i,:] == t[i+1,:])

    objective = cp.Minimize(m)
    problem = cp.Problem(objective,constraints)
    problem.solve(solver=cp.GLPK)
    return (A.value[0,:], B.value[0,:])

##### Evaluation functions #####
def plot_payout_functions(bdf,odf,a,b):
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

    plt.tight_layout()
    plt.show()

def make_eval_dfs(test_x,test_y,pred_model,strike_vals,a,b,Ks):
    col_names = ['$P(L > {})$'.format(i) for i in [70,80,90,95]]
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

##### Exploration functions #####
def parameter_exploration(data,params,model):
    model_predictions = data['model predictions']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    strike_val = data['strike val']
    loop_param = [k for k,v in params.items() if type(v) is np.ndarray][0]
    loop_param_vals = params[loop_param]
    param_subdict = {k: v for k,v in params.items() if type(v) is not np.ndarray}
    fig, axes = plt.subplots(5,1,figsize=(10,10))
    ldf = pd.DataFrame()
    for loop_param_val, ax in zip(loop_param_vals, axes.ravel()):
        loop_params = dict(list(param_subdict.items()) + [(loop_param,loop_param_val)])
        a,b = min_CVaR_program(model_predictions,train_y,loop_params)
        a,b = np.around((a,b),2)
        bdf, odf = make_eval_dfs(test_x,test_y,pred_model,strike_val,a,b,loop_params['K'])
        plot_payout_functions(ax,bdf,odf,a,b,loop_params)
        
        opt_stats = odf.drop(columns=['Losses','PredictedLosses','NetLoss']).mean().to_dict()
        opt_stats[loop_param] = str(loop_param_val)
        ldf = ldf.append(opt_stats,ignore_index=True)
        
        base_stats = bdf.drop(columns=['Losses','PredictedLosses','NetLoss']).mean().to_dict()
        base_stats[loop_param] = 'Baseline'
        ldf = ldf.append(base_stats,ignore_index=True)
                         
    fig_dir = FIGURES_DIR + '/' + model
    table_dir = TABLES_DIR + '/' + model

    plt.tight_layout()
    filename = fig_dir+'/{}_exploration.png'.format(loop_param)
    plt.savefig(filename)
    
    table_filename = table_dir +'/{}_exploration.tex'.format(loop_param)
    ldf.drop_duplicates().sort_values(loop_param).to_latex(table_filename,float_format='%.2f')

def CVaR_program_exploration(model_name='CVaR'):
    train_y, train_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=100)
    eval_y, eval_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=20)
    test_y, test_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=1000)

    pred_model = LinearRegression().fit(train_x,train_y)
    model_predictions = pred_model.predict(train_x)
    strike_p, strike_val = determine_strike_value(train_y,eval_y,eval_x,pred_model)

    data = {'train_y':train_y,'test_x':test_x,'test_y':test_y,
            'model predictions':model_predictions,'strike val':strike_val}
    
#     premium exploration
    epsilon = 0.2
    K = 20
    premiums = np.around(np.linspace(0.1,0.7,5),3)
    parameter_exploration(data,{'epsilon':epsilon,'K':K,'max_premium':premiums},model_name)
                         
#     K exploration
    Ks = np.around(np.linspace(20,40,5),3)
    epsilon = 0.2
    premium = 0.2
    parameter_exploration(data,{'epsilon':epsilon,'K':Ks,'max_premium':premium},model_name)

# Epsilon Exploration
    K = 30
    epsilons = np.around(np.linspace(0.1,0.8,5),3)
    premium = 0.3
    parameter_exploration(data,{'epsilon':epsilons,'K':K,'max_premium':premium},model_name)

beta = np.array([[1.5,0],[0,2]])
mu = np.array([3,5])
sigma = np.array(np.array([[1,1.2],[1.2,4]]))

train_y, train_x = make_multi_zone_data(beta,mu,sigma,n=100)
eval_y, eval_x = make_multi_zone_data(beta,mu,sigma,n=20)
test_y, test_x = make_multi_zone_data(beta,mu,sigma,n=1000)

pred_model = LinearRegression().fit(train_x,train_y)
pred_y = pred_model.predict(train_x)
strike_ps, strike_vals = determine_strike_values(train_y,eval_y,eval_x,pred_model)

params = {'max_premium':0.5,'min_premium':0.1,'epsilon':0.1,'epsilon_p':0.05,
         'K':np.array([10,20]),'beta':np.array([0.2,0.8]),'c_k':0.05}

a,b = np.around(min_CVaR_program(pred_y,train_y,params),2)
bdf, odf = make_eval_dfs(test_x,test_y,pred_model,strike_vals,a,b,params['K'])
plot_payout_functions(bdf,odf,a,b)

plt.close()
