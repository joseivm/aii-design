import pandas as pd
import os
import numpy as np
import sklearn.metrics as metrics
import time
import cvxpy as cp
import random, string

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
EVAL_RESULTS_DIR = os.path.join(EXPERIMENTS_DIR,'evaluation')
PREDICTIONS_DIR = os.path.join(EXPERIMENTS_DIR,'prediction','predictions')

# What am I going to do?
# 4. Figure out what to do about train, val, and test set.

##### Data Loading #####
def load_model_predictions(model_name):
    pred_file = os.path.join(PREDICTIONS_DIR,f"{model_name}_preds.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

def load_chen_payouts(market_loading,c_k,constrained):
    pred_name = f"NN Payouts ml{market_loading} ck{c_k}".replace('.','')
    pred_file = os.path.join(EVAL_RESULTS_DIR,'payouts',f"{pred_name} {constrained}.csv")
    return pd.read_csv(pred_file)

def load_payouts(model_name):
    pred_file = os.path.join(EVAL_RESULTS_DIR,'payouts',f"{model_name}.csv")
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

    # premium definition 
    constraints.append(pi == (1/n_samples)*cp.sum(alpha,axis=0) + (1/np.sum(zone_sizes)*c_k*K))
    constraints.append(max_premium >= pi)

    objective = cp.Maximize((1/n_samples)*cp.sum(-(1/risk_coef)*cp.exp(-risk_coef*(w_0 - train_y + omega - market_loading*pi))))
    problem = cp.Problem(objective,constraints)
    
    problem.solve()
    
    return (a.value, b.value)

def calculate_opt_payouts(pred_y, a, b, P):
    odf = pd.DataFrame()
    odf['PredLoss'] = pred_y
    odf['Payout'] = np.minimum(a*odf['PredLoss'] -b, P)
    odf['Payout'] = np.maximum(0, odf['Payout'])
    return odf 

def calculate_premium(payout_df, c_k): 
    payout_cvar = CVaR(payout_df,'Payout','Payout',0.01)
    average_payout = payout_df['Payout'].mean()
    required_capital = payout_cvar-average_payout
    premium = average_payout + c_k*required_capital
    return premium

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

##### Eval #####
def run_eval(model_name, params):
    # Load all predictions
    model_preds = load_model_predictions(model_name)

    # Get training preds
    train_y = model_preds.loc[model_preds.Test == False,'Loss'].to_numpy()
    train_preds = model_preds.loc[model_preds.Test == False,'PredLoss'].to_numpy()

    # Get testing preds
    test_y = model_preds.loc[model_preds.Test == True, 'Loss'].to_numpy()
    test_preds = model_preds.loc[model_preds.Test == True, 'PredLoss']

    # Design contracts
    params['P'] = np.round(train_y.max(),0)
    start = time.time()
    a, b = optimization_program(train_preds, train_y, params)
    end = time.time()
    print(f"Runtime: {(end-start)/60}")
    max_payouts = params['P']
    opt_train_payouts = calculate_opt_payouts(train_preds, a, b, max_payouts)

    opt_premium = calculate_premium(opt_train_payouts,params['c_k'])

    opt_test_payouts = calculate_opt_payouts(test_preds, a, b, max_payouts)
    opt_eval_df = create_eval_df(test_y, opt_test_payouts, opt_premium, params)
    
    results = calculate_performance_metrics(opt_eval_df, params)
    eval_name = create_eval_name(model_name, params)
    results['Eval Name'] = eval_name
    results['Model'] = model_name
    results['a'] = np.round(a[0],2)
    results['b'] = np.round(b[0],2)
    results['Market Loading'] = params['market_loading']
    results['Params'] = str(params)

    # Save results to file
    eval_df_filename = os.path.join(EVAL_RESULTS_DIR, 'payouts', f"{eval_name}.csv")
    opt_eval_df.to_csv(eval_df_filename,index=False,float_format = '%.3f')
    save_results(results, EVAL_RESULTS_DIR)
    
def run_chen_eval(constrained,params):
    chen_payouts = load_chen_payouts(params['market_loading'],params['c_k'],constrained)
    chen_train_payouts = chen_payouts.loc[chen_payouts.Set == 'Train', :]

    chen_test_payouts = chen_payouts.loc[chen_payouts.Set == 'Test', :]
    test_y = chen_payouts.loc[chen_payouts.Set == 'Test','Loss'].to_numpy()

    chen_premium = calculate_premium(chen_train_payouts, params['c_k'])
    chen_eval_df = create_eval_df(test_y, chen_test_payouts, chen_premium, params)

    chen_metrics = calculate_performance_metrics(chen_eval_df, params)
    chen_metrics['Method'] = f"Chen {constrained}"
    chen_metrics['Market Loading'] = params['market_loading']
    chen_metrics['Params'] = str(params)
    save_results(chen_metrics, EVAL_RESULTS_DIR)

def no_insurance_eval(params):
    payouts = load_chen_payouts(params['market_loading'],'constrained')
    test_payouts = payouts.loc[payouts.Test == True,:]
    test_y = test_payouts['Loss'].to_numpy()

    payout_df = pd.DataFrame()
    payout_df['Loss'] = test_y
    payout_df['Payout'] = 0

    eval_df = create_eval_df(test_y, payout_df, 0, params)
    metrics = calculate_performance_metrics(eval_df,params)
    metrics['Method'] = 'No Insurance'
    metrics['Params'] = str(params)
    save_results(metrics, EVAL_RESULTS_DIR)

def create_eval_df(test_y, payout_df, premium, params):
    edf = pd.DataFrame()
    edf['Loss'] = test_y
    edf['Payout'] = payout_df['Payout'].to_numpy()
    edf['Premium'] = premium
    edf['Wealth'] = params['w_0'] - edf['Loss'] + edf['Payout'] - params['market_loading']*edf['Premium']
    return edf

def calculate_performance_metrics(payout_df, params):
    # Utility, CEW, Payout CVaR (insurer risk), insurance cost
    pdf = payout_df
    w_0, alpha = params['w_0'], params['risk_coef']
    pdf['Utility'] = -(1/alpha)*np.exp(-alpha*pdf['Wealth'])
    average_utility = pdf['Utility'].mean()
    CEW = -np.log(-alpha*average_utility)/alpha - w_0
    insurer_risk = CVaR(payout_df,'Payout','Payout',0.01)
    return {
        'Utility': average_utility,
        'CEW': CEW,
        'Insurer Risk': insurer_risk,
        'Premium': pdf['Premium'].mean()
    }

def save_results(metrics_dict, results_dir):
    mdf = pd.DataFrame([metrics_dict])
    results_file = os.path.join(results_dir, 'results.csv')
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
    cpdf['Wealth'] = params['w_0'] - cpdf['Loss'] + cpdf['Payout'] - cpdf['Premium']

# Main Script
model_name = 'catch22_SVR_QYE'
params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,
                'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1}
# run_eval(model_name,params)
# params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,
#                 'premium_ub':200,'risk_coef':0.008,'S':1, 'market_loading':1}
# run_eval(model_name,params)
# params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,
#                 'premium_ub':300,'risk_coef':0.008,'S':1, 'market_loading':1}
# run_eval(model_name,params)

# params = {'epsilon_p':0.01,'c_k':0,'subsidy':0,'w_0':388.6,
#                 'premium_ub':100,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
# run_eval(model_name,params)
# params = {'epsilon_p':0.01,'c_k':0,'subsidy':0,'w_0':388.6,
#                 'premium_ub':200,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
# run_eval(model_name,params)
# params = {'epsilon_p':0.01,'c_k':0,'subsidy':0,'w_0':388.6,
#                 'premium_ub':300,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
# run_eval(model_name,params)
# params = {'epsilon_p':0.01,'c_k':0.15,'subsidy':0,'w_0':388.6,
#                 'premium_ub':400,'risk_coef':0.01,'S':1, 'market_loading':1.2414}
# run_eval(model_name,params)

# params = {'epsilon_p':0.01,'c_k':0.15,'subsidy':0,'w_0':388.6,
#                 'premium_ub':300,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
# no_insurance_eval(params)
# run_chen_eval('constrained',params)
# run_chen_eval('unconstrained',params)
params = {'epsilon_p':0.01,'c_k':0.09,'subsidy':0,'w_0':388.6, 'c_k':0,
                'premium_ub':300,'risk_coef':0.008,'S':1, 'market_loading':1.2414}
# no_insurance_eval(params)
run_chen_eval('constrained',params)
run_chen_eval('unconstrained',params)


