import pandas as pd
import os
import numpy as np
import time
import cvxpy as cp
import random, string
from pathlib import Path
import math
from sklearn.linear_model import LinearRegression

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
# TODO: add years to payout data, make sure it also includes the training data. 

##### Data Loading #####
def load_model_predictions(state, length, model_name):
    pred_dir = os.path.join(PREDICTIONS_DIR,state,f"predictions {length}")
    pred_file = os.path.join(pred_dir,f"{model_name}_preds.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

# def load_chen_payouts(state, length, params):
#     length = str(length)
#     market_loading,c_k = params['market_loading'], params['c_k']
#     lr, constrained = params['lr'], params['constrained']
#     payout_dir = os.path.join(EVAL_DIR,state, 'Chen Payouts')
#     pred_name = f"NN Payouts {state} L{length} ml{market_loading} ck{c_k} lr{lr}".replace('.','')
#     pred_file = os.path.join(payout_dir,f"{pred_name} {constrained}.csv")
#     return pd.read_csv(pred_file)

def load_chen_payouts(state, length, market_loading):
    length = str(length)
    if market_loading == 1 and state == 'Illinois':
        params = {'market_loading':1, 'c_k':0.13, 'lr':0.01, 'constrained':'uc'}
    elif market_loading == 1 and state in ['Iowa','Indiana','Missouri']:
        params = {'market_loading':1, 'c_k':0.13, 'lr':0.005, 'constrained':'uc'}
    else:
        params = {'market_loading':1.2414, 'c_k':0, 'lr':0.001, 'constrained':'uc'}

    market_loading,c_k = params['market_loading'], params['c_k']
    lr, constrained = params['lr'], params['constrained']
    payout_dir = os.path.join(EVAL_DIR,state, 'Chen Payouts 2')
    pred_name = [fname for fname in os.listdir(payout_dir) if f"L{length} ml{market_loading}".replace('.','') in fname][0]
    # pred_name = f"NN Payouts {state} L{length} ml{market_loading} ck{c_k} lr{lr}".replace('.','')
    # pred_file = os.path.join(payout_dir,f"{pred_name} {constrained}.csv")
    pred_file = os.path.join(payout_dir,pred_name)
    return pd.read_csv(pred_file)

def load_payouts(state, length, model_name):
    length = str(length)
    payout_dir = os.path.join(EVAL_DIR,state,'Test',f"payouts {length}")
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
    return market_loading*premium, required_capital

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

def chantarat_optimization(pred_y, train_y, eval_y):
    pred_losses = pred_y
    strike_percentiles = np.arange(0.1,0.35,0.05)
    strike_vals = np.quantile(train_y,strike_percentiles)
    strike_performance = {}

    for  strike_val in strike_vals:
        strike_val = np.around(strike_val,2)
        insured_loss = np.maximum(eval_y-strike_val,0)
        payout = np.maximum(pred_losses-strike_val,0).reshape(-1,1)
        loss_share_model = LinearRegression().fit(payout,insured_loss)
        share_explained = loss_share_model.coef_[0]
        strike_performance[str(strike_val)] = share_explained

    best_strike_val = max(strike_performance,key=strike_performance.get)
    return float(best_strike_val)

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

    opt_premium, required_capital = calculate_premium(opt_train_payouts,params['c_k'],params['market_loading'])

    opt_test_payouts = add_opt_payouts(test_df, a, b, max_payouts)
    opt_eval_df = create_eval_df(opt_test_payouts, opt_premium, params)
    opt_train_df = create_eval_df(opt_train_payouts, opt_premium, params)
    
    results = calculate_performance_metrics(opt_eval_df, params)
    eval_name = create_eval_name(model_name, params)
    results['Eval Name'] = eval_name
    results['Method'] = 'Our Method'
    results['Required Capital'] = required_capital
    results['a'] = np.round(a[0],2)
    results['b'] = np.round(b[0],2)
    results['Market Loading'] = params['market_loading']
    results['Params'] = str(params)

    # Save results to file
    if eval_set == 'Test':
        opt_eval_df['Set'] = 'Test'
        opt_train_df['Set'] = 'Train'
        opt_eval_df = pd.concat([opt_train_df,opt_eval_df],ignore_index=True)
        payout_dir = os.path.join(EVAL_DIR,state,eval_set,f"payouts {length}")
        Path(payout_dir).mkdir(exist_ok=True,parents=True)
        eval_df_filename = os.path.join(payout_dir, f"{eval_name}.csv")
        opt_eval_df.to_csv(eval_df_filename,index=False,float_format = '%.3f')
    results_dir = os.path.join(EVAL_DIR,state,eval_set)
    Path(results_dir).mkdir(exist_ok=True,parents=True)
    save_results(results, results_dir, length)
    
def run_chen_eval(state, length,params):
    length = str(length)
    chen_payouts = load_chen_payouts(state, length, params['market_loading'])
    chen_train_payouts = chen_payouts.loc[chen_payouts.Set == 'Train', :]

    chen_test_payouts = chen_payouts.loc[chen_payouts.Set == 'Test', :]

    chen_premium, req_capital = calculate_premium(chen_train_payouts, params['c_k'],params['market_loading'])
    chen_eval_df = create_eval_df(chen_test_payouts, chen_premium, params)

    chen_metrics = calculate_performance_metrics(chen_eval_df, params)
    chen_metrics['Method'] = f"Chen {params['constrained']}"
    chen_metrics['Market Loading'] = params['market_loading']
    chen_metrics['Params'] = str(params)
    chen_metrics['Required Capital'] = req_capital
    results_dir = os.path.join(EVAL_DIR,state,'Test')
    save_results(chen_metrics, results_dir, length)

def run_chantarat_eval(state, length, params, model_name, eval_set='Test'):
    length = str(length)
    # Load all predictions
    model_preds = load_model_predictions(state, length, model_name)

    # Get training preds
    train_df = model_preds.loc[model_preds.Set == 'Train',:]
    train_y = train_df['Loss'].to_numpy()
    train_preds = train_df['PredLoss'].to_numpy()

    # Get val preds
    eval_df = model_preds.loc[model_preds.Set == 'Val',:]
    eval_y = eval_df['Loss'].to_numpy()
    eval_preds = eval_df['PredLoss'].to_numpy()

    # Get testing preds
    test_df = model_preds.loc[model_preds.Set == eval_set,:]

    # Design contracts
    params['P'] = np.round(train_y.max(),0)
    b = chantarat_optimization(eval_preds, train_y, eval_y)
    max_payouts = params['P']
    train_payouts = add_opt_payouts(train_df, 1, b, max_payouts)

    premium, required_capital = calculate_premium(train_payouts,params['c_k'],params['market_loading'])

    test_payouts = add_opt_payouts(test_df, 1, b, max_payouts)
    eval_df = create_eval_df(test_payouts, premium, params)
    train_df = create_eval_df(train_payouts, premium, params)
    
    results = calculate_performance_metrics(eval_df, params)
    eval_name = create_eval_name(model_name, params)
    results['Eval Name'] = eval_name
    results['Method'] = 'Chantarat'
    results['Required Capital'] = required_capital
    results['a'] = np.round(1,2)
    results['b'] = np.round(b,2)
    results['Market Loading'] = params['market_loading']
    results['Params'] = str(params)

    # Save results to file
    if eval_set == 'Test':
        eval_df['Set'] = 'Test'
        train_df['Set'] = 'Train'
        opt_eval_df = pd.concat([train_df,eval_df],ignore_index=True)
        payout_dir = os.path.join(EVAL_DIR,state,eval_set,f"Chantarat payouts {length}")
        Path(payout_dir).mkdir(exist_ok=True,parents=True)
        eval_df_filename = os.path.join(payout_dir, f"{eval_name}.csv")
        opt_eval_df.to_csv(eval_df_filename,index=False,float_format = '%.3f')
    results_dir = os.path.join(EVAL_DIR,state,eval_set)
    Path(results_dir).mkdir(exist_ok=True,parents=True)
    save_results(results, results_dir, length)

def no_insurance_eval(state, length, params):
    length = str(length)
    payouts = load_chen_payouts(state, length, params['market_loading'])
    test_payouts = payouts.loc[payouts.Set == 'Test',:]
    test_y = test_payouts['Loss'].to_numpy()

    payout_df = pd.DataFrame()
    payout_df['Loss'] = test_y
    payout_df['Payout'] = 0

    eval_df = create_eval_df(payout_df, 0, params)
    metrics = calculate_performance_metrics(eval_df,params)
    metrics['Method'] = 'No Insurance'
    metrics['Params'] = str(params)
    results_dir = os.path.join(EVAL_DIR,state,'Test')
    save_results(metrics, results_dir, length)

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
        'Premium': pdf['Premium'].mean(),
        'Insurer Cost': pdf['Payout'].sum()
    }

def save_results(metrics_dict, results_dir, length):
    mdf = pd.DataFrame([metrics_dict])
    results_file = os.path.join(results_dir, f"results_{length}_2.csv")
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

def get_best_model(state, length, market_loading, method='Our Method'):
    length = str(length)
    pred_dir = os.path.join(EVAL_DIR,state,'Val')
    results_fname = os.path.join(pred_dir,f"results_{length}_2.csv")
    rdf = pd.read_csv(results_fname)
    rdf = rdf.loc[rdf['Market Loading'] == market_loading,:]
    rdf = rdf.loc[rdf['Method'] == method,:]
    # rdf = rdf.loc[rdf['Eval Name'].str.contains('chen'),:]
    idx = rdf['Utility'].idxmax()
    best_model = rdf.loc[idx, 'Eval Name']
    return '_'.join(best_model.split('_')[:3])

def get_eval_name(state, length, market_loading):
    length = str(length)
    pred_dir = os.path.join(EVAL_DIR,state,'Test')
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    rdf = rdf.loc[(rdf['Market Loading'] == market_loading) & (rdf['Method'] == 'Our Method'),:]
    idx = rdf['Utility'].idxmax()
    best_model = rdf.loc[idx, 'Eval Name']
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

def get_premium(state,market_loading, length):
    fname = os.path.join(EVAL_DIR,state,'Test',f"results_{length}_2.csv")
    rdf = pd.read_csv(fname)
    premium = rdf.loc[(rdf['Market Loading'] == market_loading) & (rdf.Method == 'Chen uc'),'Premium'].item()
    return math.ceil(premium)

def choose_best_model(state, length, params):
    length = str(length)
    pred_dir = os.path.join(PREDICTIONS_DIR,state)
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    bad_model_dict = {str(i*10):[] for i in range(2,9)}
    bad_model_dict['83'] = []
    # bad_models = []
    for model_name in rdf['Model Name'].unique():
        print(model_name)
        bad_models = bad_model_dict[length]
        model_prefix = '_'.join(model_name.split('_')[:2])
        if model_prefix not in bad_models and 'chen' in model_prefix: 
            try:
                run_eval(state, length, params, model_name, 'Val')
            except cp.error.SolverError:
                print(f"{model_name} failed to run")
                pass

def choose_best_model_chantarat(state, length, params):
    length = str(length)
    pred_dir = os.path.join(PREDICTIONS_DIR,state)
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    for model_name in rdf['Model Name'].unique():
        print(model_name)
        try:
            run_chantarat_eval(state, length, params, model_name, 'Val')
        except cp.error.SolverError:
            print(f"{model_name} failed to run")
            pass

# Main Script
state = 'Illinois'
# lengths = [i*10 for i in range(3,9)] 
lengths = [40]
state_init_w_0 = {'Illinois':913-504+388.6, 'Indiana':818-504+388.6, 'Iowa':879-504+388.6,
                  'Missouri':873-504+388.6}
# We got these by calculating maximum revenue across all time for each state, it's the maximum
# value of "TS Value"*3.5 in all of the data, the 504 is the cost of operating the farm, which is
# 504 according to the Chen replication package. 
for length in lengths:
    print(length)
    ##### Our definition of the premium #####
    # premium_ub = 200
    premium_ub = get_premium(state,1,length)+1
    params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':state_init_w_0[state],
                'premium_ub':premium_ub,'risk_coef':0.008,'S':1, 'market_loading':1}
    # choose_best_model(state, length, params)
    model_name = get_best_model(state, length, 1)
    run_eval(state, length, params, model_name)
    # choose_best_model_chantarat(state,length,params)
    # model_name = get_best_model(state, length, 1, method='Chantarat')
    # run_chantarat_eval(state, length,params,model_name)



    # params['constrained'] = 'uc'
    # run_chen_eval(state, length ,params)

    ##### Their definition of the premium #####
    # premium_ub = get_premium(1.241,length)
    # premium_ub = 100
    # params = {'epsilon_p':0.01,'c_k':0,'subsidy':0,'w_0':388.6,
    #                 'premium_ub':premium_ub,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
    # choose_best_model(state,length, params)
    # model_name = get_best_model(state, length, 1.241)
    # run_eval(state, length, params, model_name)

    # params = {'epsilon_p':0.01,'subsidy':0,'w_0':state_init_w_0[state], 'c_k':0,'lr':0.001,'constrained':'uc',
    #                 'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
    # run_chen_eval(state, length, params)
    # premium_ub = get_premium(state,1.241,length)
    # choose_best_model(state, length, params)
    # model_name = get_best_model(state, length, 1.241)
    # run_eval(state, length, params, model_name)
    # no_insurance_eval(state, length, params)





# state = 'Illinois'
# market_loading = 1
# best_model_40 = get_best_model(state, 40, market_loading)
# best_model_50 = get_best_model(state, 50, market_loading)

# gdf = load_model_predictions(state, 40, best_model_40)
# bdf = load_model_predictions(state, 50, best_model_50)
    
# best_model_60 = get_eval_name(state, 60, 1)
# df60 = load_payouts(state, 60, best_model_60)