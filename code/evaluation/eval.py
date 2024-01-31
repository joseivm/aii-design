import pandas as pd
import os
import numpy as np
import sklearn.metrics as metrics
import time
import cvxpy as cp
import random, string
from pathlib import Path
import math

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
EVAL_DIR = os.path.join(EXPERIMENTS_DIR,'evaluation')
PREDICTIONS_DIR = os.path.join(EXPERIMENTS_DIR,'prediction')

# What am I going to do?
# TODO: update to handle different lengths
# 4. Figure out what to do about train, val, and test set.

##### Data Loading #####
def load_model_predictions(state, length, model_name):
    pred_dir = os.path.join(PREDICTIONS_DIR,state,f"predictions {length}")
    pred_file = os.path.join(pred_dir,f"{model_name}_preds.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

def load_chen_payouts(state, length, params):
    length = str(length)
    market_loading,c_k = params['market_loading'], params['c_k']
    lr, constrained = params['lr'], params['constrained']
    payout_dir = os.path.join(EVAL_DIR,state, 'Chen Payouts')
    pred_name = f"NN Payouts {state} L{length} ml{market_loading} ck{c_k} lr{lr}".replace('.','')
    pred_file = os.path.join(payout_dir,f"{pred_name} {constrained}.csv")
    return pd.read_csv(pred_file)

def load_payouts(state, length, model_name):
    length = str(length)
    payout_dir = os.path.join(EVAL_DIR,state,length,'payouts')
    pred_file = os.path.join(payout_dir,f"{model_name}.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

##### Contract Design #####
def optimization_program(pred_y,train_y,params):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    # max_premium, max_payouts
    eps_p = params['epsilon_p']
    zone_sizes = params['S']
    c_k = params['c_k']
    max_premium = params['premium_ub']
    max_payout = params['P']
    market_loading = params['market_loading']
    w_0 = params['w_0']
    risk_coef = params['risk_coef']

    if pred_y.ndim == 1:
        pred_y = pred_y[:,np.newaxis]
        train_y = train_y[:,np.newaxis]

    n_samples = train_y.shape[0]
    n_zones = train_y.shape[1]
    S = np.tile(zone_sizes,(n_samples,1))
    p = np.ones(n_samples)/n_samples

        # contract vars
    a = cp.Variable(n_zones)
    b = cp.Variable(n_zones)
    pi = cp.Variable(n_zones)

    # cvar vars
    t_k = cp.Variable()
    gamma_p = cp.Variable(n_samples)

    # approximation vars
    alpha = cp.Variable((n_samples,n_zones))
    omega = cp.Variable((n_samples,n_zones))

    K = cp.Variable()

    constraints = []

    # objective, m >= CVaR(l_z - I_z(theta_z))
    # constraints.append(t + (1/epsilon)*(p @ gamma) <= m*np.ones(n_zones)) # 0

    # CVaR constraints for each zone's loss, gamma^k_z >= l_z - min(a_z \hat{l_z}+b_z, K_z) - t
    # constraints.append(gamma >= train_y + premium_discount*cp.vstack([pi]*n_samples) - omega - cp.vstack([t]*n_samples)) # 1
    # constraints.append(gamma >= 0) # 2

    # Portfolio CVaR constraint: CVaR(\sum_z I_z(\theta_z)) <= K^P + \sum_z E[I_z(\theta_Z)]
    constraints.append(t_k + (1/eps_p)*(p @ gamma_p) <= K + (1/n_samples)*cp.sum(cp.multiply(S,omega))) # 5
    constraints.append(gamma_p >= cp.sum(cp.multiply(S,alpha),axis=1)- cp.reshape(cp.vstack([t_k]*n_samples),(n_samples,))) # 6
    constraints.append(gamma_p >= 0) # 7
    constraints.append(omega <= cp.multiply(pred_y,cp.vstack([a]*n_samples))-cp.vstack([b]*n_samples)) # 8
    constraints.append(omega <= cp.vstack([max_payout]*n_samples)) # 9 
    constraints.append(alpha >= cp.multiply(pred_y,cp.vstack([a]*n_samples))-cp.vstack([b]*n_samples)) # 10
    constraints.append(alpha >= 0) # 11

    # budget constraint
    # constraints.append(max_premium >= (1/n_samples)*cp.sum(cp.multiply(S,alpha)) + c_k*K) # 12
    constraints.append(b >= 0)

    # premium definition 
    constraints.append(pi == (1/n_samples)*cp.sum(alpha,axis=0) + (1/np.sum(zone_sizes)*c_k*K))
    constraints.append(max_premium >= market_loading*pi)

    objective = cp.Maximize((1/n_samples)*cp.sum(-(1/risk_coef)*cp.exp(-risk_coef*(w_0 - train_y + omega - market_loading*pi))))
    problem = cp.Problem(objective,constraints)
    
    problem.solve()
    
    return (a.value, b.value)

def add_opt_payouts(odf, a, b, P):
    odf = odf.copy()
    odf['Payout'] = np.minimum(a*odf['PredLoss'] -b, P)
    odf['Payout'] = np.maximum(0, odf['Payout'])
    return odf 

def calculate_premium(payout_df, c_k, market_loading): 
    payout_cvar = CVaR(payout_df,'Payout','Payout',0.01)
    average_payout = payout_df['Payout'].mean()
    required_capital = payout_cvar-average_payout
    premium = average_payout + c_k*required_capital
    return market_loading*premium

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

##### Eval #####
def run_eval(state, length, params, model_name, eval_set='Test'):
    length = str(length)
    # Load all predictions
    model_preds = load_model_predictions(state, length, model_name)

    # Get training preds
    train_df = model_preds.loc[model_preds.Set == 'Train',:]
    train_y = train_df['Loss'].to_numpy()
    train_preds = train_df['PredLoss'].to_numpy()

    # Get testing preds
    test_df = model_preds.loc[model_preds.Set == eval_set,:]

    # Design contracts
    params['P'] = np.round(train_y.max(),0)
    start = time.time()
    a, b = optimization_program(train_preds, train_y, params)
    end = time.time()
    print(f"Runtime: {(end-start)/60}")
    max_payouts = params['P']
    opt_train_payouts = add_opt_payouts(train_df, a, b, max_payouts)

    opt_premium = calculate_premium(opt_train_payouts,params['c_k'],params['market_loading'])

    opt_test_payouts = add_opt_payouts(test_df, a, b, max_payouts)
    opt_eval_df = create_eval_df(opt_test_payouts, opt_premium, params)
    
    results = calculate_performance_metrics(opt_eval_df, params)
    eval_name = create_eval_name(model_name, params)
    results['Eval Name'] = eval_name
    results['Method'] = 'Our Method'
    results['a'] = np.round(a[0],2)
    results['b'] = np.round(b[0],2)
    results['Market Loading'] = params['market_loading']
    results['Params'] = str(params)

    # Save results to file
    if eval_set == 'Test':
        payout_dir = os.path.join(EVAL_DIR,state,eval_set,f"payouts {length}")
        Path(payout_dir).mkdir(exist_ok=True,parents=True)
        eval_df_filename = os.path.join(payout_dir, f"{eval_name}.csv")
        opt_eval_df.to_csv(eval_df_filename,index=False,float_format = '%.3f')
    results_dir = os.path.join(EVAL_DIR,state,eval_set)
    Path(results_dir).mkdir(exist_ok=True,parents=True)
    save_results(results, results_dir, length)
    
def run_chen_eval(state, length,params):
    length = str(length)
    chen_payouts = load_chen_payouts(state, length, params)
    chen_train_payouts = chen_payouts.loc[chen_payouts.Set == 'Train', :]

    chen_test_payouts = chen_payouts.loc[chen_payouts.Set == 'Test', :]

    chen_premium = calculate_premium(chen_train_payouts, params['c_k'],params['market_loading'])
    chen_eval_df = create_eval_df(chen_test_payouts, chen_premium, params)

    chen_metrics = calculate_performance_metrics(chen_eval_df, params)
    chen_metrics['Method'] = f"Chen {params['constrained']}"
    chen_metrics['Market Loading'] = params['market_loading']
    chen_metrics['Params'] = str(params)
    results_dir = os.path.join(EVAL_DIR,state,length)
    save_results(chen_metrics, results_dir)

def no_insurance_eval(state, length, params):
    length = str(length)
    payouts = load_chen_payouts(state, length, params)
    test_payouts = payouts.loc[payouts.Set == 'Test',:]
    test_y = test_payouts['Loss'].to_numpy()

    payout_df = pd.DataFrame()
    payout_df['Loss'] = test_y
    payout_df['Payout'] = 0

    eval_df = create_eval_df(payout_df, 0, params)
    metrics = calculate_performance_metrics(eval_df,params)
    metrics['Method'] = 'No Insurance'
    metrics['Params'] = str(params)
    results_dir = os.path.join(EVAL_DIR,state,length)
    save_results(metrics, results_dir)

def create_eval_df(payout_df, premium, params):
    w_0, alpha = params['w_0'], params['risk_coef']
    edf = payout_df.copy()
    edf['Premium'] = premium
    edf['Wealth'] = w_0 - edf['Loss'] + edf['Payout'] - edf['Premium']
    edf['Utility'] = -(1/alpha)*np.exp(-alpha*edf['Wealth'])
    return edf

def calculate_performance_metrics(payout_df, params):
    # Utility, CEW, Payout CVaR (insurer risk), insurance cost
    # TODO: insurer risk should be based on training data, maybe turn it into cost
    pdf = payout_df.copy()
    alpha = params['risk_coef']
    average_utility = pdf['Utility'].mean()
    CEW = -np.log(-alpha*average_utility)/alpha
    insurer_risk = CVaR(payout_df,'Payout','Payout',0.01)
    return {
        'Utility': average_utility,
        'CEW': CEW,
        'Insurer Risk': insurer_risk,
        'Premium': pdf['Premium'].mean()
    }

def save_results(metrics_dict, results_dir, length):
    mdf = pd.DataFrame([metrics_dict])
    results_file = os.path.join(results_dir, f"results_{length}.csv")
    if os.path.isfile(results_file):
        rdf = pd.read_csv(results_file)
    else: 
        rdf = pd.DataFrame()
    rdf = pd.concat([rdf,mdf],ignore_index=True)
    rdf.to_csv(results_file,float_format = '%.3f',index=False)

def create_eval_name(model_name, params):
    eval_name = f"{model_name}_ub{params['premium_ub']}_"
    model_id = ''.join(random.choices(string.ascii_letters + string.digits,k=3))
    return eval_name + model_id

def debug():
    # Their premium analysis
    params = {'epsilon_p':0.01,'c_k':0,'subsidy':0,'w_0':388.6,
                    'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
    constrained = 'unconstrained'
    chen_payouts = load_chen_payouts(params['market_loading'],params['c_k'],constrained)
    chen_train_payouts = chen_payouts.loc[chen_payouts.Set == 'Train', :]

    chen_test_payouts = chen_payouts.loc[chen_payouts.Set == 'Test', :]

    chen_premium = calculate_premium(chen_train_payouts, params['c_k'],params['market_loading'])
    chen_eval_df = create_eval_df(chen_payouts, chen_premium, params)
    cdf = chen_eval_df.copy()

    # our premium analysis
    params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,
                'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1}
    constrained = 'constrained'
    chen_payouts = load_chen_payouts(params['market_loading'],params['c_k'],constrained)
    chen_train_payouts = chen_payouts.loc[chen_payouts.Set == 'Train', :]

    chen_test_payouts = chen_payouts.loc[chen_payouts.Set == 'Test', :]

    chen_premium = calculate_premium(chen_train_payouts, params['c_k'],params['market_loading'])
    chen_eval_df = create_eval_df(chen_payouts, chen_premium, params)
    odf = chen_eval_df.copy()

def get_best_model(state, length):
    length = str(length)
    pred_dir = os.path.join(PREDICTIONS_DIR,state,length)
    results_fname = os.path.join(pred_dir,'results.csv')
    rdf = pd.read_csv(results_fname)
    rdf['F1'] = 2*rdf['Loss Recall']*rdf['Payout Precision']/(rdf['Loss Recall'] + rdf['Payout Precision'])
    idx = rdf['F1'].idxmax()
    best_model = rdf.loc[idx, 'Model Name']
    return best_model

def get_results():
    lengths = [i*10 for i in range(3,10)]
    rdfs = []
    for length in lengths:

        fname = os.path.join(EVAL_DIR,'Illinois',str(length),'results.csv')
        rdf = pd.read_csv(fname)
        rdf['Length'] = length
        rdfs.append(rdf)

    rdf = pd.concat(rdfs)

    ni_df = rdf.loc[rdf.Method == 'No Insurance',['Length','Utility']]
    rdf = rdf.merge(ni_df,on='Length',suffixes=('',' NI'))
    rdf['UtilityImprovement'] = (rdf['Utility']-rdf['Utility NI'])/rdf['Utility NI']
    tst = rdf.groupby(['Length','Market Loading','Method'])['UtilityImprovement'].max().reset_index()

def get_premium(market_loading, length):
    fname = os.path.join(EVAL_DIR,'Illinois',str(length),'results.csv')
    rdf = pd.read_csv(fname)
    premium = rdf.loc[(rdf['Market Loading'] == market_loading) & (rdf.Method == 'Chen uc'),'Premium'].item()
    return math.ceil(premium)

def choose_best_model(state, length, params):
    length = str(length)
    pred_dir = os.path.join(PREDICTIONS_DIR,state)
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    bad_models = []
    # bad_models = []
    for model_name in rdf['Model Name'].unique():
        print(model_name)
        model_prefix = '_'.join(model_name.split('_')[:2])
        if model_prefix not in bad_models:
            run_eval(state, length, params, model_name, 'Val')

# Main Script
state = 'Illinois'
lengths = [i*10 for i in range(3,8)] + [83]
# lengths = [98]
for length in lengths:
    print(length)
    ##### Our definition of the premium #####
    # premium_ub = get_premium(1,length)
    params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,
                    'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1}
    choose_best_model(state, length, params)

    # params['premium_ub'] = 65
    # run_eval(state, length, params)

    # params['premium_ub'] = 100
    # run_eval(state, length, params)

    # params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,'lr':0.01,'constrained':'c',
    #                 'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1}
    # run_chen_eval(state, length, params)

    # params['constrained'] = 'uc'
    # run_chen_eval(state, length ,params)

    ##### Their definition of the premium #####
    # premium_ub = get_premium(1.241,length)
    # params = {'epsilon_p':0.01,'c_k':0,'subsidy':0,'w_0':388.6,
    #                 'premium_ub':premium_ub,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
    # run_eval(state, length,params)

    # params['premium_ub'] = 65
    # run_eval(state, length, params)

    # params['premium_ub'] = 100
    # run_eval(state, length, params)

    # params = {'epsilon_p':0.01,'subsidy':0,'w_0':388.6, 'c_k':0,'lr':0.001,'constrained':'c',
    #                 'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
    # no_insurance_eval(state, length, params)
    # run_chen_eval(state, length, params)
    # params['constrained'] = 'uc'
    # run_chen_eval(state, length, params)
# params = {'epsilon_p':0.01,'c_k':0,'subsidy':0,'w_0':388.6,
#                 'premium_ub':200,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
# run_eval(model_name,params)
# params = {'epsilon_p':0.01,'c_k':0,'subsidy':0,'w_0':388.6,
#                 'premium_ub':300,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
# run_eval(model_name,params)
# params = {'epsilon_p':0.01,'c_k':0.15,'subsidy':0,'w_0':388.6,
#                 'premium_ub':400,'risk_coef':0.01,'S':1, 'market_loading':1.2414}
# run_eval(model_name,params)

# params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,'lr':0.01,
#                 'premium_ub':300,'risk_coef':0.008,'S':1, 'market_loading':1}
# no_insurance_eval(params)
# run_chen_eval('constrained',params)
# run_chen_eval('unconstrained',params)
# params = {'epsilon_p':0.01,'subsidy':0,'w_0':388.6, 'c_k':0,'lr':0.001,
#                 'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
# no_insurance_eval(params)
# run_chen_eval('constrained',params)
# run_chen_eval('unconstrained',params)


