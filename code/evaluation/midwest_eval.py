import pandas as pd
import os
import numpy as np
import time
import cvxpy as cp
import random, string
from pathlib import Path
from functools import reduce
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
TRANSFORMS_DIR = os.path.join(PROJECT_DIR,'data','time-series-transforms')

##### Data Loading #####
def load_model_predictions(state, length):
    model_name = get_best_model(state, length, 1)
    pred_dir = os.path.join(PREDICTIONS_DIR,state,f"predictions {length}")
    pred_file = os.path.join(pred_dir,f"{model_name}_preds.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

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
    payout_dir = os.path.join(EVAL_DIR,state, 'Chen Payouts')
    pred_name = f"NN Payouts {state} L{length} ml{market_loading} ck{c_k} lr{lr}".replace('.','')
    pred_file = os.path.join(payout_dir,f"{pred_name} {constrained}.csv")
    return pd.read_csv(pred_file)

def load_chantarat_payouts(state, length):
    results_fname = os.path.join(EVAL_DIR,state,'Test',f"results_{length}_2.csv")
    rdf = pd.read_csv(results_fname)
    model_name = rdf.loc[rdf.Method == 'Chantarat','Eval Name'].item()
    payout_fname = os.path.join(EVAL_DIR, state, 'Test', f"Chantarat payouts {length}",f"{model_name}.csv")
    payouts = pd.read_csv(payout_fname)
    return payouts 

def load_payouts(state, length, model_name):
    length = str(length)
    payout_dir = os.path.join(EVAL_DIR,state,'Test',f"payouts {length}")
    pred_file = os.path.join(payout_dir,f"{model_name}.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

def load_years(state, length,val=True):
    state_dir = os.path.join(TRANSFORMS_DIR,state)
    train_fname = os.path.join(state_dir,f"train_years_L{length}.npy")
    val_fname = os.path.join(state_dir,f"val_years_L{length}.npy")
    test_fname = os.path.join(state_dir,f"test_years_L{length}.npy")

    train_yrs = np.load(train_fname,allow_pickle=True)
    val_yrs = np.load(val_fname,allow_pickle=True)
    test_yrs = np.load(test_fname,allow_pickle=True)
    if val:
        yrs = np.concatenate([train_yrs, val_yrs, test_yrs])
    else:
        yrs = np.concatenate([train_yrs,test_yrs])
    return yrs

def get_best_model(state, length, market_loading):
    length = str(length)
    pred_dir = os.path.join(EVAL_DIR,state,'Val')
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    rdf = rdf.loc[rdf['Market Loading'] == market_loading,:]
    rdf = rdf.loc[rdf['Eval Name'].str.contains('chen'),:]
    idx = rdf['Utility'].idxmax()
    best_model = rdf.loc[idx, 'Eval Name']
    return '_'.join(best_model.split('_')[:3])

def create_multi_state_data(states, length):
    # TODO: consider using statified sampling like in Thailand
    length = str(length)
    # Load all predictions
    train_dfs = {}
    for state in states:
        model_preds = load_model_predictions(state, length)
        pred_years = load_years(state, length)
        model_preds['State'] = state
        model_preds['CountyYear'] = pred_years
        model_preds['Year'] = model_preds['CountyYear'].apply(lambda x: int(x.split('-')[1]))
        train_dfs[state] = model_preds.loc[model_preds.Set == 'Train',:]

    min_train_state = min(train_dfs.items(), key = lambda x: x[1]['Year'].nunique())[0]

    all_year_dfs = []
    for year in train_dfs[min_train_state]['Year'].unique():
        year_dfs = []
        largest_state = max(train_dfs.items(), key = lambda x: x[1].loc[x[1].Year == year].shape[0])[0]
        big_df = train_dfs[largest_state]
        year_size = big_df.loc[big_df.Year == year,:].shape[0]
        for state in states:
            sdf = train_dfs[state]
            sdf = sdf.loc[sdf.Year == year,['Loss','PredLoss']].sample(year_size, replace=True, random_state=5)
            sdf.rename(columns={'Loss':f"Loss_{state}",'PredLoss':f"PredLoss_{state}"},inplace=True)
            sdf.reset_index(inplace=True,drop=True)
            year_dfs.append(sdf)

        ydf = pd.concat(year_dfs,axis=1)
        ydf['Year'] = year
        all_year_dfs.append(ydf)

    train_df = pd.concat(all_year_dfs,ignore_index=True)
    return train_df

def load_all_model_predictions(states, length):
    length = str(length)
    dfs = []
    for state in states:
        model_preds = load_model_predictions(state, length)
        pred_years = load_years(state, length)
        model_preds['State'] = state
        model_preds['CountyYear'] = pred_years
        model_preds['Year'] = model_preds['CountyYear'].apply(lambda x: int(x.split('-')[1]))
        dfs.append(model_preds)

    return pd.concat(dfs,ignore_index=True)
    
def load_all_chen_payouts(states, length):
    length = str(length)
    all_dfs = []
    for state in states:
        chen_payouts = load_chen_payouts(state, length, 1)
        pred_years = load_years(state, length)
        chen_payouts['CountyYear'] = pred_years
        chen_payouts['Year'] = chen_payouts['CountyYear'].apply(lambda x: int(x.split('-')[1]))
        chen_payouts['State'] = state
        all_dfs.append(chen_payouts)

    return pd.concat(all_dfs, ignore_index=True)

def load_all_chantarat_payouts(states, length):
    length = str(length)
    all_dfs = []
    for state in states:
        chen_payouts = load_chantarat_payouts(state, length)
        pred_years = load_years(state, length,False)
        chen_payouts['CountyYear'] = pred_years
        chen_payouts['Year'] = chen_payouts['CountyYear'].apply(lambda x: int(x.split('-')[1]))
        chen_payouts['State'] = state
        all_dfs.append(chen_payouts)

    return pd.concat(all_dfs, ignore_index=True)


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

    objective = cp.Maximize((1/n_samples)*cp.sum(-(1/risk_coef)*cp.exp(-risk_coef*(w_0 - train_y + omega - market_loading*cp.vstack([pi]*n_samples)))))
    problem = cp.Problem(objective,constraints)
    
    problem.solve()
    
    return (a.value, b.value)

def add_opt_payouts(df, a, b, P):
    sdfs = []
    for state in a.keys():
        sdf = df.loc[df.State == state,:].copy()
        sdf['Payout'] = np.minimum(a[state]*sdf['PredLoss'] -b[state],P[state])
        sdf['Payout'] = np.maximum(0, sdf['Payout'])
        sdfs.append(sdf)
    
    return pd.concat(sdfs,ignore_index=True)

def calculate_premiums(df, c_k, state_sizes):
    state_years = [df.loc[df.State == state,'Year'].unique() for state in df.State.unique()]
    years = reduce(lambda x,y: np.intersect1d(x,y), state_years)
    total_payouts = []
    for year in years: 
        total_payout = df.loc[df.Year == year,'Payout'].sum()
        total_payouts.append({'Year':year,'TotalPayout':total_payout})

    pdf = pd.DataFrame(total_payouts)

    payout_cvar = CVaR(pdf, 'TotalPayout', 'TotalPayout', 0.01)
    average_payout = pdf['TotalPayout'].mean()
    required_capital = payout_cvar - average_payout

    premiums = {}
    for state in df.State.unique():
        premiums[state] = df.loc[df.State == state,'Payout'].mean() + c_k*required_capital/np.sum(state_sizes)
         
    return premiums, required_capital

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

##### Eval #####
def run_eval(states, length, params, eval_set='Test'):
    length = str(length)

    # Load all predictions
    ms_train_df = create_multi_state_data(states, length)
    model_preds = load_all_model_predictions(states, length)

    # Get training preds
    loss_cols = [f"Loss_{state}" for state in states]
    pred_cols = [f"PredLoss_{state}" for state in states]
    train_y = ms_train_df.loc[:,loss_cols].to_numpy()
    train_preds = ms_train_df.loc[:,pred_cols].to_numpy()
    train_df = model_preds.loc[model_preds.Set == 'Train',:]

    # Design contracts
    params['P'] = np.round(train_y.max(axis=0),0)
    params['premium_ub'] = [get_premium(state,params['market_loading'],length) for state in states]
    start = time.time()
    a, b = optimization_program(train_preds, train_y, params)
    end = time.time()
    print(f"Runtime: {(end-start)/60}")
    a = {state: val for state, val in zip(states, a)}
    b = {state: val for state, val in zip(states, b)}

    # Get test_preds
    test_df = model_preds.loc[model_preds.Set == eval_set,:]

    max_payouts = {state: P for state, P in zip(states, params['P'])}
    opt_train_payouts = add_opt_payouts(train_df, a, b, max_payouts)

    opt_premiums, req_capital = calculate_premiums(opt_train_payouts,params['c_k'],params['S'],)

    # This can also take a list/dict of dfs
    opt_test_payouts = add_opt_payouts(test_df, a, b, max_payouts)
    opt_eval_df = create_eval_df(opt_test_payouts, opt_premiums, params)
    opt_train_df = create_eval_df(opt_train_payouts, opt_premiums, params)

    results = calculate_performance_metrics(opt_eval_df)
    results['Required Capital'] = req_capital
    eval_name = create_eval_name(states, params)
    results['Eval Name'] = eval_name
    results['Method'] = 'Our Method'
    for state in states:
        results[f"{state}_a"] = np.round(a[state],2)
        results[f"{state}_b"] = np.round(b[state],2)
        # results['Params'] = str(params)

    # Save results to file
    if eval_set == 'Test':
        # opt_eval_df['Set'] = 'Test'
        # opt_train_df['Set'] = 'Train'
        opt_eval_df = pd.concat([opt_train_df,opt_eval_df],ignore_index=True)
        payout_dir = os.path.join(EVAL_DIR,'Midwest',eval_set,f"payouts {length}")
        Path(payout_dir).mkdir(exist_ok=True,parents=True)
        eval_df_filename = os.path.join(payout_dir, f"{eval_name}.csv")
        opt_eval_df.to_csv(eval_df_filename,index=False,float_format = '%.3f')
    
    params['Eval Name'] = eval_name
    results_dir = os.path.join(EVAL_DIR,'Midwest',eval_set)
    Path(results_dir).mkdir(exist_ok=True,parents=True)
    save_results(results, results_dir, length)
    save_params(params, results_dir, length)
    
def run_chen_eval(states, length, params):
    chen_payouts = load_all_chen_payouts(states, length)
    train_payouts = chen_payouts.loc[chen_payouts.Set == 'Train',:]
    test_payouts = chen_payouts.loc[chen_payouts.Set == 'Test',:]

    chen_premiums, req_capital = calculate_premiums(train_payouts, params['c_k'],params['S'])
    chen_eval_df = create_eval_df(test_payouts, chen_premiums, params)

    chen_metrics = calculate_performance_metrics(chen_eval_df)
    chen_metrics['Method'] = f"Chen uc"
    chen_metrics['Required Capital'] = req_capital
    # chen_metrics['Params'] = str(params)
    params['Eval Name'] = chen_metrics['Method']
    results_dir = os.path.join(EVAL_DIR,'Midwest','Test')
    save_results(chen_metrics, results_dir, length)
    save_params(params, results_dir, length)

def run_chantarat_eval(states, length, params):
    payouts = load_all_chantarat_payouts(states, length)
    train_payouts = payouts.loc[payouts.Set == 'Train',:]
    test_payouts = payouts.loc[payouts.Set == 'Test',:]

    premiums, req_capital = calculate_premiums(train_payouts, params['c_k'], params['S'])
    eval_df = create_eval_df(test_payouts, premiums, params)

    metrics = calculate_performance_metrics(eval_df)
    metrics['Method'] = 'Chantarat'
    metrics['Required Capital'] = req_capital
    params['Eval Name'] = metrics['Method']
    results_dir = os.path.join(EVAL_DIR,'Midwest','Test')
    save_results(metrics, results_dir, length)
    save_params(params, results_dir, length)


def no_insurance_eval(states, length, params):
    length = str(length)
    pdfs = []
    for state in states:
        payouts = load_chen_payouts(state, length, 1)
        payouts['State'] = state
        pdfs.append(payouts)

    pdf = pd.concat(pdfs, ignore_index=True)

    test_payouts = pdf.loc[pdf.Set == 'Test',:]
    test_y = test_payouts['Loss'].to_numpy()

    payout_df = pd.DataFrame()
    payout_df['Loss'] = test_y
    payout_df['Payout'] = 0
    payout_df['State'] = test_payouts['State'].to_numpy()
    premiums = {state: 0 for state in states}

    eval_df = create_eval_df(payout_df, premiums, params)
    metrics = calculate_performance_metrics(eval_df)
    metrics['Method'] = 'No Insurance'
    params['Method'] = 'No Insurance'
    results_dir = os.path.join(EVAL_DIR,'Midwest','Test')
    save_results(metrics, results_dir, length)
    save_params(params,results_dir, length)

def create_eval_df(payout_df, premiums, params):
    edfs = []
    w_0, alpha = params['w_0'], params['risk_coef']
    for state in premiums.keys():
        edf = payout_df.loc[payout_df.State == state,:].copy()
        edf['Premium'] = premiums[state]
        edf['Wealth'] = w_0 - edf['Loss'] + edf['Payout'] - edf['Premium']
        edf['Utility'] = -(1/alpha)*np.exp(-alpha*edf['Wealth'])
        edfs.append(edf)

    return pd.concat(edfs,ignore_index=True)

def calculate_performance_metrics(payout_df):
    bdf = payout_df.copy()
    average_utility = bdf['Utility'].mean()
    results = {}
    results['Overall Utility'] = average_utility
    for state in payout_df.State.unique():
        results[f"{state}_Premium"] = bdf.loc[bdf.State == state,'Premium'].mean()
        results[f"{state}_Utility"] = bdf.loc[bdf.State == state,'Utility'].mean()

    results['Insurer Cost'] = bdf.Payout.sum()
    return results

def save_results(metrics_dict, results_dir, length):
    mdf = pd.DataFrame([metrics_dict])
    results_file = os.path.join(results_dir, f"results_{length}.csv")
    if os.path.isfile(results_file):
        rdf = pd.read_csv(results_file)
    else: 
        rdf = pd.DataFrame()
    rdf = pd.concat([rdf,mdf],ignore_index=True)
    rdf.to_csv(results_file,float_format = '%.3f',index=False)

def save_params(params, results_dir, length):
    mdf = pd.DataFrame([params])
    results_file = os.path.join(results_dir, f"params_{length}.csv")
    if os.path.isfile(results_file):
        rdf = pd.read_csv(results_file)
    else: 
        rdf = pd.DataFrame()
    rdf = pd.concat([rdf,mdf],ignore_index=True)
    rdf.to_csv(results_file,float_format = '%.3f',index=False)

def create_eval_name(states, params):
    state_abbrv = {'Illinois':'IL','Indiana':'IN','Iowa':'IA','Missouri':'MO'}
    states = [state_abbrv[state] for state in states]
    eval_name = f"{'_'.join(states)}_ml{params['market_loading']}_"
    model_id = ''.join(random.choices(string.ascii_letters + string.digits,k=3))
    return eval_name + model_id

def get_best_model(state, length, market_loading):
    length = str(length)
    pred_dir = os.path.join(EVAL_DIR,state,'Val')
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    rdf = rdf.loc[rdf['Market Loading'] == market_loading,:]
    rdf = rdf.loc[rdf['Eval Name'].str.contains('chen'),:]
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
    fname = os.path.join(EVAL_DIR,state,'Test',f"results_{length}.csv")
    rdf = pd.read_csv(fname)
    premium = rdf.loc[(rdf['Market Loading'] == market_loading) & (rdf.Method == 'Chen uc'),'Premium'].item()
    return math.ceil(premium)


# Main Script
states = ['Illinois','Indiana','Iowa','Missouri']
lengths = [i*10 for i in range(2,9)]
# lengths = [20,30]
for length in lengths:
    print(length)
    ##### Our definition of the premium #####
    # premium_ub = get_premium(1,length)
    # consider giving it the same premium_ub as the chen models. 
    params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,
            'risk_coef':0.008,'S':[87,74,94,46], 'market_loading':1}
    # choose_best_model(state, length, params)
    # model_name = get_best_model(state, length, 1)
    # run_eval(states, length, params)
    run_chantarat_eval(states, length,params)
    # run_chen_eval(states,length,params)
    # no_insurance_eval(states, length, params)

    # params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,'lr':0.01,'constrained':'uc',
                    # 'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1}
    # run_chen_eval(state, length, params)

    ##### Their definition of the premium #####
    # premium_ub = get_premium(1.241,length)
    # premium_ub = 100
    # params = {'epsilon_p':0.01,'c_k':0,'subsidy':0,'w_0':388.6,
    #                 'premium_ub':premium_ub,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
    # choose_best_model(state,length, params)
    # model_name = get_best_model(state, length, 1.241)
    # run_eval(state, length, params, model_name)


    # params = {'epsilon_p':0.01,'subsidy':0,'w_0':388.6, 'c_k':0,'lr':0.001,'constrained':'uc',
    #                 'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
    # no_insurance_eval(state, length, params)
    # run_chen_eval(state, length, params)
    # params['constrained'] = 'uc'
    # run_chen_eval(state, length, params)

