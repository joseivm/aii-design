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
def make_single_zone_data(beta,mu,sigma,n=100):
    theta = np.random.normal(loc=mu,scale=sigma,size=n)
    epsilon = np.random.normal(loc=0,scale=1)
    w = beta*theta + epsilon
    return w, theta.reshape(-1,1)

##### Optimization Functions #####
def min_premium_program(pred_y,train_y,loss_p,epsilon,model='full',a_lb=0):
    w_bar = np.quantile(train_y,loss_p)
    n_samples = train_y.shape[0]
    I = cp.Variable(n_samples)
    a = cp.Variable()
    b = cp.Variable()
    t = cp.Variable()
    gamma = cp.Variable(n_samples)

    constraints = []
    constraints.append(I >= 0)
    constraints.append(t + (1/epsilon)*(1/n_samples)*(sum(gamma))<= w_bar)
    constraints.append(gamma >=  pred_y + (1/n_samples)*sum(I) - I -t)
    constraints.append(gamma >= 0)
    
    if model == 'full':
        constraints.append(I >= a*pred_y + b)
        constraints.append(b <= 0)
        constraints.append(a >= a_lb)
    elif model == 'slope':
        constraints.append(I >= a*pred_y)
        constraints.append(a >= a_lb)
    else:
        constraints.append(I >= pred_y + b)
        constraints.append(b <= 0)

    objective = cp.Minimize(sum(I))
    problem = cp.Problem(objective,constraints)
    problem.solve()
    print(problem.status)
    return_dict = {'full':(a.value,b.value),'slope':(a.value,0),'intercept':(1,b.value)}
    return return_dict[model]

def min_CVaR_program(pred_y,train_y,params):
    max_pi = params['max_premium']
    K = params['K']
    epsilon = params['epsilon']

    n_samples = train_y.shape[0]
    a = cp.Variable()
    b = cp.Variable()
    t = cp.Variable()
    gamma = cp.Variable(n_samples)

    constraints = []
    constraints.append(a*pred_y+b >= 0)
    constraints.append(a*pred_y+b >= -gamma + train_y -t)
    constraints.append(K >= -gamma + train_y -t)
    constraints.append(gamma >= 0)
    constraints.append(a*(1/n_samples)*sum(pred_y) + b <= K*max_pi)

    objective = cp.Minimize(t + (1/epsilon)*(1/n_samples)*(sum(gamma)))
    problem = cp.Problem(objective,constraints)
    problem.solve(solver=cp.GLPK)
    return (a.value, b.value)

##### Evaluation functions #####
def plot_payout_functions(ax,bdf,odf,a,b,args):
    ax.plot(bdf['PredictedLosses'],bdf['Premium'],'bs',label='baseline')
    ax.plot(odf['PredictedLosses'],odf['Premium'],'g^',label='opt')
    ax.plot(odf['PredictedLosses'],odf['Losses'],'ro',label='actual losses')
    ax.set_title(str(args))
    ax.text(3,15,'a = {}, b = {}'.format(a,b))
    ax.legend()

def make_eval_dfs(test_x,test_y,pred_model,strike_val,a,b,K):
    loss_quantiles = np.quantile(test_y,[0.7,0.8,0.9,0.95])
    col_names = ['$P(L > {})$'.format(i) for i in [70,80,90,95]]
    
    bdf = pd.DataFrame()
    odf = pd.DataFrame()
    bdf['Losses'] = test_y
    bdf['PredictedLosses'] = pred_model.predict(test_x)
    bdf['Payout'] = np.maximum(bdf['PredictedLosses']-strike_val,0)
    bdf['NetLoss'] = bdf['Losses'] - bdf['Payout']
    
    odf['Losses'] = test_y
    odf['PredictedLosses'] = pred_model.predict(test_x)
    odf['Payout'] = np.minimum(a*odf['PredictedLosses']+b,K)
    odf['NetLoss'] = odf['Losses']-odf['Payout']
    
    for col_name, loss_quantile in zip(col_names,loss_quantiles):
        bdf[col_name] = bdf['NetLoss'] > loss_quantile
        odf[col_name] = odf['NetLoss'] > loss_quantile
    odf = odf.rename(columns={'Payout':'Premium'})
    bdf = bdf.rename(columns={'Payout':'Premium'})
    return bdf, odf

def determine_strike_value(train_y,eval_y,eval_x,pred_model):
    strike_percentiles = np.arange(0.1,0.35,0.05)
    strike_vals = np.quantile(train_y,strike_percentiles)
    
    pred_losses = pred_model.predict(eval_x)
    strike_performance = {}

    for strike_percentile, strike_val in zip(strike_percentiles,strike_vals):
        strike_percentile = np.around(strike_percentile,2)
        insured_loss = np.maximum(eval_y-strike_val,0)
        payout = np.maximum(pred_losses-strike_val,0).reshape(-1,1)
        loss_share_model = LinearRegression().fit(payout,insured_loss)
        share_explained = loss_share_model.coef_[0]
        strike_performance[(strike_percentile,strike_val)] = share_explained

    best_strike_percentile, best_strike_val = max(strike_performance,key=strike_performance.get)
    return best_strike_percentile, best_strike_val

##### Exploration functions #####

def parameter_exploration(data,params,model):
    model_predictions = data['model predictions']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    strike_val = data['strike val']
    pred_model = data['pred_model']
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

def premium_program_exploration(model):
    train_y, train_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=100)
    eval_y, eval_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=20)
    test_y, test_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=1000)

    pred_model = LinearRegression().fit(train_x,train_y)
    model_predictions = pred_model.predict(train_x)
    strike_p, strike_val = determine_strike_value(train_y,eval_y,eval_x,pred_model)
    
    fig_dir = FIGURES_DIR + '/Premium Model'
    table_dir = TABLES_DIR + '/Premium Model'
    
#     loss_quantile exploration
    epsilon = 0.2
    loss_quantiles = np.around(np.linspace(0.55,0.9,5),3)
    fig, axes = plt.subplots(5,1,figsize=(10,10))
    ldf = pd.DataFrame()
    for loss_q,ax in zip(loss_quantiles,axes.ravel()):
        a,b = min_premium_program(model_predictions,train_y,loss_q,epsilon,model)
        a,b = np.around((a,b),2)
        bdf, odf = make_eval_dfs(test_x,test_y,pred_model,strike_val,a,b)
        plot_payout_functions2(ax,bdf,odf,a,b,loss_q,epsilon)
        
        opt_stats = odf.drop(columns=['Losses','PredictedLosses','NetLoss']).mean().to_dict()
        opt_stats['Loss q'] = str(loss_q)
        ldf = ldf.append(opt_stats,ignore_index=True)
        
        base_stats = bdf.drop(columns=['Losses','PredictedLosses','NetLoss']).mean().to_dict()
        base_stats['Loss q'] = 'Baseline'
        ldf = ldf.append(base_stats,ignore_index=True)
                         
    plt.tight_layout()
    filename = fig_dir+'/loss_q_exploration_{}.png'.format(model)
    plt.savefig(filename)
    
    table_filename = table_dir +'/loss_q_exploration_{}.tex'.format(model)
    ldf.drop_duplicates().sort_values('Loss q').to_latex(table_filename,float_format='%.2f')
                         
#     epsilon exploration
    epsilons = np.linspace(0.05,0.5,5)
    loss_quantile = 0.6
    fig, axes = plt.subplots(5,1,figsize=(10,10))
    ldf = pd.DataFrame()
    for eps,ax in zip(epsilons,axes.ravel()):
        a,b = min_premium_program(model_predictions,train_y,loss_quantile,eps,model)
        a,b = np.around((a,b),2)
        bdf, odf = make_eval_dfs(test_x,test_y,pred_model,strike_val,a,b)
        plot_payout_functions2(ax,bdf,odf,a,b,loss_quantile,eps)
        
        opt_stats = odf.drop(columns=['Losses','PredictedLosses','NetLoss']).mean().to_dict()
        opt_stats['Eps'] = str(eps)
        ldf = ldf.append(opt_stats,ignore_index=True)
        
        base_stats = bdf.drop(columns=['Losses','PredictedLosses','NetLoss']).mean().to_dict()
        base_stats['Eps'] = 'Baseline'
        ldf = ldf.append(base_stats,ignore_index=True)
                         
    plt.tight_layout()
    filename = fig_dir+'/epsilon_exploration_{}.png'.format(model)
    plt.savefig(filename)
    
    table_filename = table_dir +'/epsilon_exploration_{}.tex'.format(model)
    ldf.drop_duplicates().sort_values('Eps').to_latex(table_filename,float_format='%.2f')

def CVaR_program_exploration(model_name='CVaR'):
    train_y, train_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=100)
    eval_y, eval_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=20)
    test_y, test_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=1000)

    pred_model = LinearRegression().fit(train_x,train_y)
    model_predictions = pred_model.predict(train_x)
    strike_p, strike_val = determine_strike_value(train_y,eval_y,eval_x,pred_model)

    inputs = {'train_y':train_y,'test_x':test_x,'test_y':test_y,'pred_model':pred_model,
            'model predictions':model_predictions,'strike val':strike_val}
    
#     premium exploration
    epsilon = 0.2
    K = 20
    premiums = np.around(np.linspace(0.1,0.7,5),3)
    parameter_exploration(inputs,{'epsilon':epsilon,'K':K,'max_premium':premiums},model_name)
                         
#     K exploration
    Ks = np.around(np.linspace(20,40,5),3)
    epsilon = 0.2
    premium = 0.2
    parameter_exploration(inputs,{'epsilon':epsilon,'K':Ks,'max_premium':premium},model_name)

# Epsilon Exploration
    K = 30
    epsilons = np.around(np.linspace(0.1,0.8,5),3)
    premium = 0.3
    parameter_exploration(inputs,{'epsilon':epsilons,'K':K,'max_premium':premium},model_name)

CVaR_program_exploration('CVaR3')
# np.random.seed(1)
# train_y, train_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=100)
# eval_y, eval_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=20)
# test_y, test_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=1000)

# pred_model = LinearRegression().fit(train_x,train_y)
# pred_y = pred_model.predict(train_x)
# strike_p, strike_val = determine_strike_value(train_y,eval_y,eval_x,pred_model)

# epsilon = 0.1
# max_pi = 0.5
# K = 3000
# plt.close()
# a,b = np.around(min_CVaR_program(pred_y,train_y,{'max_premium':max_pi,'K':K,'epsilon':epsilon}),2)
# print('a = {}, b = {}'.format(a,b))
# edf = pd.DataFrame()
# edf['PredictedLoss'] = pred_y
# edf['OptPayout'] = np.minimum(a*edf['PredictedLoss']+b,K)
# edf['BasePayout'] = np.maximum(edf['PredictedLoss']-strike_val,0)
# edf['Actual Loss'] = train_y

# plt.plot(edf['PredictedLoss'],edf['BasePayout'],'bs',label='baseline')
# plt.plot(edf['PredictedLoss'],edf['OptPayout'],'g^',label='opt')
# plt.plot(edf['PredictedLoss'],edf['Actual Loss'],'ro',label='loss')
# plt.legend()
# plt.title('epsilon: {}, pi: {}'.format(epsilon,max_pi))
# plt.text(3,10,'a = {}, b = {}'.format(a,b))
# plt.xlabel(r'$\hat{l}(\theta^k)$')
# plt.ylabel('Payout, I')
# plt.show()
# plt.savefig('tst.png')
# plt.close()

# plt.plot(edf['PredictedLosses'],edf['BaselinePayout'],'bs',label='baseline')
# plt.plot(edf['PredictedLosses'],edf['OptPayout'],'g^',label='opt')
# plt.legend()
# plt.show()
# plt.close()