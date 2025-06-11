import pandas as pd
import os
import sys
sys.path.insert(0,'/Users/joseivelarde/Projects/aii-design/code/evaluation')
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import cvxpy as cp
import copy
from pathlib import Path
import itertools
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from thai_synthetic_data import simulate_single_zone_payouts

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
DATA_DIR = os.path.join(PROJECT_DIR,'data')

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR,'evaluation','Thailand')
PREDICTIONS_DIR = os.path.join(EXPERIMENTS_DIR,'prediction','Thailand')
MODELS_DIR = os.path.join(PREDICTIONS_DIR, 'model-preds')

# Global Variables
ZONES = ['C1','C2','C3','N1','N2','N3','NE1','NE2','NE3','S1','S2']
TEST_YEARS = np.arange(2015, 2023)

##### Data Loading #####
def load_model_predictions(model_name, zone, test_year):
    pred_dir = os.path.join(PREDICTIONS_DIR,'model-preds',model_name)
    pred_file = os.path.join(pred_dir,f"{zone}_{test_year}_preds.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

def load_loss_data(zone):
    fpath = os.path.join(DATA_DIR,'processed','Thailand_loss_data.csv')
    df = pd.read_csv(fpath)
    df = df.loc[df.Zone == zone,:]
    return df.set_index('ObsID')

def stratified_loss_sample(
    df: pd.DataFrame,
    n_samples: int,
    loss_col: str = 'Loss',
    random_state: int = None
) -> pd.DataFrame:
    """
    Stratified sampling of df into zero-loss and positive-loss strata, returning sample weights.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column `loss_col`.
    n_samples : int
        Total number of rows to draw (with replacement).
    loss_col : str
        Name of the loss column to stratify on.
    strata_weights : dict, optional
        Mapping 'zero' and 'positive' to fraction of n_samples. If None, sample equally.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Stratified sample of df, length ~ n_samples, with an added 'w' column
        for the inverse-probability weight.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)

    # 1) define two strata: zero vs positive
    df['stratum'] = np.where(df[loss_col] > 0, 'positive', 'zero')

    # population count per stratum
    pop_counts = df['stratum'].value_counts().to_dict()
    strata = ['zero', 'positive']

    # 2) determine how many to draw per stratum
    # equal sample size in each stratum
    base = n_samples // 2
    counts = {'zero': base, 'positive': base}
    # assign remainder to 'zero'
    if n_samples % 2:
        counts['zero'] += 1

    # 3) sample from each stratum
    samples = []
    for s in strata:
        grp = df[df['stratum'] == s]
        if grp.empty or counts[s] == 0:
            continue
        choice = grp.sample(n=counts[s], replace=True, random_state=rng.bit_generator)
        samples.append(choice)

    result = pd.concat(samples, ignore_index=True)

    # 4) compute inverse-probability weights
    weights = {s: pop_counts.get(s, 0) / counts[s] for s in counts if counts[s] > 0}
    result['w'] = result['stratum'].map(weights)

    # 5) cleanup
    result = result.drop(columns=['stratum'])
    return result

##### Contract Design #####
def optimization_program(simulated_preds: np.ndarray, sample_df: pd.DataFrame, params: dict):
    
    """
    simulated_preds:   (n_sim x Z) array of simulated predicted zone-loss shares for premium & CVaR
    sample_df: DataFrame with columns ['Zone','Loss','w'], stratified sample for objective
    params:   {
        'epsilon_p', 'premium_ub', 'market_loading',
        'w_0', 'risk_coef', 'c_k', 'S'  # zone_sizes as list/array length Z
    }
    """
    # Unpack params
    eps_p          = params['epsilon_p']
    max_premium    = params['premium_ub']
    market_loading = params['market_loading']
    w_0            = params['w_0']
    rho            = params['risk_coef']
    c_k            = params['c_k']
    zone_sizes     = np.array(params['S'])
    sumS           = zone_sizes.sum()
    max_payout     = 1.0

    # Zones and shapes
    zones       = sorted(sample_df['Zone'].unique())
    Z           = len(zones)
    n_sim, Z2   = simulated_preds.shape
    assert Z2 == Z, "pred_y must have same #zones as sample_df"

    # ----------------------------------------------------------------------------
    # 1) SIMULATED DATA BLOCK (premium & CVaR)
    # ----------------------------------------------------------------------------
    # Decision vars
    a         = cp.Variable(Z)            # slope per zone
    b         = cp.Variable(Z)            # intercept per zone
    a_row = cp.reshape(a, (1, Z))
    b_row = cp.reshape(b, (1, Z))

    pi        = cp.Variable(Z)            # premium per zone
    t_k       = cp.Variable()             # CVaR auxiliary
    gamma     = cp.Variable(n_sim)        # CVaR auxiliary per sim
    alpha_sim = cp.Variable((n_sim, Z))   # spliced lower payouts per sim
    omega_sim = cp.Variable((n_sim, Z))   # spliced upper payouts per sim
    K         = cp.Variable()             # required capital

    # Exposures
    S_sim = np.tile(zone_sizes, (n_sim,1))   # (n_sim x Z)
    p_sim = np.ones(n_sim)/n_sim

    constraints = []

    # CVaR constraint: t + 1/eps_p * E[gamma] <= K + E[zone_size*omega]
    constraints += [
        t_k + (1/eps_p)*(p_sim @ gamma)
          <= K + (1/n_sim)*cp.sum(cp.multiply(S_sim, omega_sim))
    ]
    # gamma ≥ portfolio loss – t_k
    constraints += [
        gamma >= cp.sum(cp.multiply(S_sim, alpha_sim), axis=1) - t_k,
        gamma >= 0
    ]

    # spliced definitions on simulated pred_y
    constraints += [
        omega_sim <= cp.multiply(simulated_preds, a_row) - b_row,
        omega_sim <= max_payout,
        alpha_sim >= cp.multiply(simulated_preds, a_row) - b_row,
        alpha_sim >= 0
    ]

    # premium definition & caps
    constraints += [
        pi == (1/n_sim)*cp.sum(alpha_sim, axis=0)
              + (c_k/sumS)*K,
        max_premium >= market_loading*pi,
        b >= 0
    ]

    # ----------------------------------------------------------------------------
    # 2) OBJECTIVE BLOCK (sample_df with CARA utility)
    # ----------------------------------------------------------------------------
    # We’ll build one alpha_obj[z], omega_obj[z] per zone
    alpha_obj = {}
    omega_obj = {}

    # Prepare per-zone sample arrays
    zone_to_pred_y = {z: grp['PredLoss'].to_numpy()
                      for z, grp in sample_df.groupby('Zone')}
    zone_to_y = {z: grp['Loss'].to_numpy()
                 for z, grp in sample_df.groupby('Zone')}
    zone_to_w = {z: grp['w'].to_numpy()
                 for z, grp in sample_df.groupby('Zone')}

    obj_terms = []
    for iz, z in enumerate(zones):
        pred_y_z = zone_to_pred_y[z]
        y_z = zone_to_y[z]   # shape (n_z,)
        w_z = zone_to_w[z]   # shape (n_z,)
        n_z = len(y_z)

        # per-zone decision vars
        alpha_obj[z] = cp.Variable(n_z)
        omega_obj[z] = cp.Variable(n_z)

        # spliced constraints on actual data
        constraints += [
            omega_obj[z] <= a[iz] * pred_y_z - b[iz],
            omega_obj[z] <= max_payout,
            alpha_obj[z] >= a[iz] * pred_y_z - b[iz],
            alpha_obj[z] >= 0
        ]

        # CARA payoff: w0 + 1 - loss + payout - loading * pi_z
        payoff_z = w_0 + 1 - y_z + omega_obj[z] - market_loading*pi[iz]
        util_z   = (1/(1-rho)) * cp.power(payoff_z, 1-rho)

        # weighted & size‐weighted zone contribution
        sum_wz    = w_z.sum()
        zone_term = zone_sizes[iz] * (w_z @ util_z) / sum_wz
        obj_terms.append(zone_term)

    # ----------------------------------------------------------------------------
    # 3) OBJECTIVE & SOLVE
    # ----------------------------------------------------------------------------
    objective = cp.Maximize(cp.sum(obj_terms))

    prob = cp.Problem(objective, constraints)
    prob.solve(max_iter=1000)

    return a.value[0], b.value[0]

def optimization_program_old(pred_y,train_y,params):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    # max_premium, max_payouts
    eps_p = params['epsilon_p']
    zone_sizes = params['S']
    c_k = params['c_k']
    max_premium = params['premium_ub']
    max_payout = 1
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
    constraints.append(gamma_p >= cp.sum(cp.multiply(S,alpha),axis=1) - t_k) # 6
    constraints.append(gamma_p >= 0) # 7
    constraints.append(omega <= cp.multiply(pred_y, a.T) - b.T) # 8
    constraints.append(omega <= max_payout) # 9 
    constraints.append(alpha >= cp.multiply(pred_y, a.T) - b.T) # 10
    constraints.append(alpha >= 0) # 11

    # premium definition and constraints
    constraints.append(pi == (1/n_samples)*cp.sum(alpha,axis=0) + (1/np.sum(zone_sizes)*c_k*K))
    constraints.append(max_premium >= market_loading*pi)
    constraints.append(b >= 0)

    objective = cp.Maximize((1/n_samples)*cp.sum((1/(1-risk_coef))*(w_0 + 1 - train_y + omega - market_loading*pi.T)**(1-risk_coef)))
    problem = cp.Problem(objective,constraints)
    
    problem.solve()
    
    return (a.value, b.value)

def chantarat_optimization(pred_y, eval_y):
    pred_losses = pred_y
    strike_vals = np.arange(0,0.35,0.05)
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

def add_opt_payouts(odf, a, b):
    odf = odf.copy()
    odf['Payout'] = np.minimum(a*odf['PredLoss'] - b, 1)
    odf['Payout'] = np.maximum(0, odf['Payout'])
    return odf 

def calculate_premium(payout_df, c_k, market_loading): 
    payout_cvar = CVaR(payout_df,'Payout','Payout',0.01)
    average_payout = payout_df['Payout'].mean()
    required_capital = payout_cvar-average_payout
    premium = average_payout + c_k*required_capital
    return market_loading*premium, required_capital

def calculate_sz_premium(payout_df, c_k, req_capital_df=None, subsidy=0): 
    if req_capital_df is None:
        req_capital_df = payout_df.copy()

    payout_df = payout_df.copy()
    payout_cvar = CVaR(req_capital_df,'Payout','Payout',0.01)
    average_payout = req_capital_df.Payout.mean()
    required_capital = payout_cvar-average_payout
    cost_of_capital = c_k*required_capital
    premium = payout_df.Payout.mean() + cost_of_capital
    return (1-subsidy)*premium, cost_of_capital

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

##### Cross-Validation #####
def design_contracts(method, params):
    if method == 'VMX':
        design_VMX_contracts(params)
    else:
        design_chantarat_contracts(params)

def design_VMX_contracts(params):
    zone, test_year = params['zone'], params['test_year']
    model_name, eval_set = params['model_name'], params['eval_set']

    # Load all predictions
    model_preds = load_model_predictions(model_name, zone, test_year)
    ldf = load_loss_data(zone)
    model_preds = model_preds.loc[model_preds.Idx.isin(ldf.index),:]

    # Get training preds
    if eval_set == 'Test':
        train_df = model_preds.loc[model_preds.Set == 'Val',:]
        test_df = model_preds.loc[model_preds.Set == 'Test',:]
        
    else:
        train_df = model_preds.loc[model_preds.Set == 'Val',:]
        train_df, test_df = train_test_split(train_df, test_size=0.5, random_state=2,stratify=train_df['Loss'] > 0)

    sample_df = stratified_loss_sample(train_df, len(train_df))
    sim_preds = simulate_single_zone_payouts(train_df, zone, n_sim=2000)
    sim_matrix = sim_preds['PredLoss'].to_numpy().reshape(-1,1)

    # Design contracts
    a, b = optimization_program(sim_matrix, sample_df, params)
    opt_train_payouts = add_opt_payouts(train_df, a, b)
    sim_train_payouts = add_opt_payouts(sim_preds,a,b)
    opt_test_payouts = add_opt_payouts(test_df, a, b)

    opt_test_premium, test_required_capital = calculate_sz_premium(opt_test_payouts,params['c_k'], sim_train_payouts)
    opt_train_premium, train_required_capital = calculate_sz_premium(opt_train_payouts,params['c_k'], sim_train_payouts)

    opt_eval_df = create_eval_df(opt_test_payouts, opt_train_premium, params['w_0'], params['risk_coef'])
    opt_train_df = create_eval_df(opt_train_payouts, opt_train_premium, params['w_0'], params['risk_coef'])

    # Save results to file
    if eval_set == 'Test':
        opt_eval_df['Set'] = 'Test'
        opt_train_df['Set'] = 'Train'
        opt_eval_df = pd.concat([opt_train_df,opt_eval_df],ignore_index=True)
        opt_eval_df['Zone'] = zone
        payout_dir = os.path.join(RESULTS_DIR, eval_set, 'payouts', f"VMX ck{params['c_k']} w{params['w_0']} r{params['risk_coef']}".replace('.',''))
        Path(payout_dir).mkdir(exist_ok=True,parents=True)
        eval_df_filename = os.path.join(payout_dir, f"{zone}_{test_year}.csv")
        opt_eval_df.to_csv(eval_df_filename,index=False,float_format = '%.3f')
    
    else:
        results = performance_metrics(opt_eval_df, params['w_0'])
        eval_name = create_eval_name('VMX', model_name, params)
        results['EvalName'] = eval_name
        results['Method'] = 'VMX'
        results['Required Capital'] = train_required_capital
        results['a'] = np.round(a,2)
        results['b'] = np.round(b,2)
        results['TestYear'] = test_year
        results['c_k'] = params['c_k']
        results['alpha'] = params['risk_coef']
        results['Zone'] = zone
        results['EvalSet'] = eval_set
        save_results_mp(results)
    
def design_chantarat_contracts(params):
    zone, test_year = params['zone'], params['test_year']
    model_name, eval_set = params['model_name'], params['eval_set']

    # Load all predictions
    model_preds = load_model_predictions(model_name, zone, test_year)
    ldf = load_loss_data(zone)
    model_preds = model_preds.loc[model_preds.Idx.isin(ldf.index),:]

    # Get training preds
    train_sets = ['Train']
    if eval_set == 'Test':
        train_sets.append('Val')

    train_df = model_preds.loc[model_preds.Set.isin(train_sets),:]
    sim_preds = simulate_single_zone_payouts(train_df, zone, n_sim=2000)
    val_df = model_preds.loc[model_preds.Set == 'Val',:]
    train_y = train_df['Loss'].to_numpy()
    train_preds = train_df['PredLoss'].to_numpy()

    # Get val preds
    val_y = val_df['Loss'].to_numpy()
    val_preds = val_df['PredLoss'].to_numpy()

    # Get testing preds
    test_df = model_preds.loc[model_preds.Set == eval_set,:]

    # Design contracts
    b = chantarat_optimization(val_preds, val_y)
    opt_train_payouts = add_opt_payouts(train_df, 1, b)
    sim_train_payouts = add_opt_payouts(sim_preds, 1, b)
    opt_test_payouts = add_opt_payouts(test_df, 1, b)

    opt_premium, required_capital = calculate_sz_premium(opt_train_payouts,params['c_k'],sim_train_payouts)

    opt_eval_df = create_eval_df(opt_test_payouts, opt_premium, params['w_0'], params['risk_coef'])
    opt_train_df = create_eval_df(opt_train_payouts, opt_premium, params['w_0'], params['risk_coef'])


    # Save results to file
    if eval_set == 'Test':
        opt_eval_df['Set'] = 'Test'
        opt_train_df['Set'] = 'Train'
        opt_eval_df = pd.concat([opt_train_df,opt_eval_df],ignore_index=True)
        opt_eval_df['Zone'] = zone
        payout_dir = os.path.join(RESULTS_DIR, eval_set, 'payouts', f"Chantarat ck{params['c_k']} w{params['w_0']} r{params['risk_coef']}".replace('.',''))
        Path(payout_dir).mkdir(exist_ok=True,parents=True)
        eval_df_filename = os.path.join(payout_dir, f"{zone}_{test_year}.csv")
        opt_eval_df.to_csv(eval_df_filename,index=False,float_format = '%.3f')
    
    else:
        results = performance_metrics(opt_eval_df, params['w_0'])
        eval_name = create_eval_name('Chantarat', model_name, params)
        results['EvalName'] = eval_name
        results['Method'] = 'Chantarat'
        results['Required Capital'] = required_capital
        results['a'] = 1
        results['b'] = np.round(b,2)
        results['TestYear'] = test_year
        results['c_k'] = params['c_k']
        results['alpha'] = params['risk_coef']
        results['Zone'] = zone
        results['EvalSet'] = eval_set
        save_results_mp(results)

def create_eval_df(payout_df, premium, w_0=0.5, alpha=1.5):
    edf = payout_df.copy()
    edf['Premium'] = premium
    edf['Wealth'] = w_0 + 1 - edf['Loss'] + edf['Payout'] - edf['Premium']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    return edf

def performance_metrics(payout_df, w_0=0.5):
    edf = payout_df.copy()

    ce_ii = certainty_equivalent(edf.Loss, edf.Payout, edf.Premium.mean(), w_0=w_0)
    ce_ii_og = certainty_equivalent(edf.Loss, edf.PredLoss, edf.PredLoss.mean(), w_0=w_0)
    ce_ni = certainty_equivalent(edf.Loss, 0, 0, w_0=w_0)
    ce_pi = certainty_equivalent(edf.Loss, edf.Loss, edf.Loss.mean(), w_0=w_0)
    
    delta_ce = 100*(ce_ii - ce_ni)/ce_ni
    max_delta_ce = 100*(ce_pi - ce_ni)/ce_ni
    rib = np.nan if max_delta_ce == 0 else delta_ce/max_delta_ce

    utility_ii = CRRA_utility(edf.Loss, edf.Payout, edf.Premium.mean(), w_0=w_0)
    utility_ii_og = CRRA_utility(edf.Loss, edf.PredLoss, edf.PredLoss.mean(), w_0=w_0)
    utility_ni = CRRA_utility(edf.Loss, 0, 0, w_0=w_0)
    utility_pi = CRRA_utility(edf.Loss, edf.Loss, edf.Loss.mean(), w_0=w_0)

    better_off = pct_better_off(edf.Loss, edf.Payout, edf.Premium.mean(), w_0=w_0)
    max_better_off = pct_better_off(edf.Loss, edf.Loss, edf.Loss.mean(), w_0=w_0)

    metrics_dict = {
        'DeltaU': 100*(utility_ii - utility_ni)/np.abs(utility_ni),
        'MaxDeltaU': 100*(utility_pi - utility_ni)/np.abs(utility_ni),
        'U_II': utility_ii,
        'U_NI': utility_ni,
        'U_II_OG': utility_ii_og,
        'U_PI': utility_pi,
        'DeltaCE': delta_ce,
        'MaxDeltaCE': max_delta_ce,
        'RIB': rib,
        'CE_II': ce_ii,
        'CE_NI': ce_ni,
        'CE_II_OG': ce_ii_og,
        'CE_PI': ce_pi,
        'BetterOff': better_off,
        'MaxBetterOff': max_better_off,
        'Premium': edf['Premium'].mean(),
        'Cost_II': edf['Payout'].mean(),
        'Cost_PI': edf['Loss'].mean(),
        'Size' : len(edf),
        'w_0' : w_0
    }
    return metrics_dict

def certainty_equivalent(y_true, y_pred, premium, w_0=0.5, alpha=1.5, markup=0):
    # TODO: change to utility
    edf = pd.DataFrame({'Loss':y_true, 'Payout':y_pred})
    edf['Wealth'] = w_0 - (1+markup)*premium + 1 - edf['Loss'] + edf['Payout']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    certainty_equivalent = ((1-alpha)*edf['Utility'].mean())**(1/(1-alpha))
    return certainty_equivalent

def CRRA_utility(y_true, y_pred, premium, w_0=0.5, alpha=1.5, markup=0):
    edf = pd.DataFrame({'Loss':y_true, 'Payout':y_pred})
    edf['Wealth'] = w_0 - (1+markup)*premium + 1 - edf['Loss'] + edf['Payout']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    return edf['Utility'].mean()

def pct_better_off(y_true, y_pred, premium, w_0=0.5, alpha=1.5, markup=0):
    edf = pd.DataFrame({'Loss':y_true, 'Payout':y_pred})
    edf['Wealth'] = w_0 - (1+markup)*premium + 1 - edf['Loss'] + edf['Payout']
    edf['Wealth_NI'] = w_0 + 1 - edf['Loss']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    edf['Utility_NI'] = 1/(1-alpha)*edf['Wealth_NI']**(1-alpha)
    edf['BetterOff'] = edf['Utility'] > edf['Utility_NI']
    return edf['BetterOff'].mean()

def create_eval_name(cd_method, model_name, params):
    eval_name = f"{cd_method}_{model_name}_ck{params['c_k']}_w{params['w_0']}_r{params['risk_coef']}"
    return eval_name.replace('.','')

def get_eval_name(state, length, market_loading):
    length = str(length)
    pred_dir = os.path.join(RESULTS_DIR,state,'Test')
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

        fname = os.path.join(RESULTS_DIR,'Illinois',str(length),'results.csv')
        rdf = pd.read_csv(fname)
        rdf['Length'] = length
        rdfs.append(rdf)

    rdf = pd.concat(rdfs)

    ni_df = rdf.loc[rdf.Method == 'No Insurance',['Length','Utility']]
    rdf = rdf.merge(ni_df,on='Length',suffixes=('',' NI'))
    rdf['UtilityImprovement'] = (rdf['Utility']-rdf['Utility NI'])/rdf['Utility NI']
    tst = rdf.groupby(['Length','Market Loading','Method'])['UtilityImprovement'].max().reset_index()

##### Generate Payouts #####
def generate_payouts(c_k, w_0, alpha, method):
    # zones = ['C1','C2','C3','N1','N2','N3','NE1','NE2','NE3','S1','S2']
    zones = ['N2','N3','NE1','NE2','NE3']
    test_years = np.arange(2015,2023)
    param_dicts = []
    for zone in zones:
        zone_model = get_best_model(zone, c_k, w_0, alpha, method)
        for year in test_years:
            param_dicts.append(create_param_dict(zone_model, zone, year, 'Test', w_0, alpha, c_k))

    for params in tqdm(param_dicts):
        design_contracts(method, params)

def get_best_model(zone, c_k, w_0, alpha, method='VMX'):
    res_dir = os.path.join(RESULTS_DIR,'Val','full-results')
    results_fname = os.path.join(res_dir,f"{zone}.csv")
    rdf = pd.read_csv(results_fname)
    rdf = rdf.loc[(rdf['c_k'] == c_k) & (rdf.Zone == zone) & (rdf.w_0 == w_0) & (rdf.alpha == alpha),:]
    rdf = rdf.loc[rdf['Method'] == method,:]
    rdf = rdf.groupby('EvalName')['DeltaCE'].mean().reset_index(name='DeltaCE')
    idx = rdf['DeltaCE'].idxmax()
    best_model = rdf.loc[idx, 'EvalName']
    return '_'.join(best_model.split('_')[1:-3])

##### Saving functions #####
def save_results_mp(metrics_dict):
    outdir = os.path.join(RESULTS_DIR,metrics_dict['EvalSet'],'intermediate-results')
    eval_name = metrics_dict['EvalName']
    zone = metrics_dict['Zone']
    test_year = metrics_dict['TestYear']
    mdf = pd.DataFrame([metrics_dict])
    mdf['Combo'] = list(zip(mdf['EvalName'],mdf['Zone'],mdf['TestYear']))
    fname = f"{eval_name}_{zone}_{test_year}".replace('.','')
    fpath = os.path.join(outdir,f"{fname}.csv")
    mdf.to_csv(fpath, float_format = '%.3f',index=False)

def save_results(metrics_dict, results_dir):
    mdf = pd.DataFrame([metrics_dict])
    results_file = os.path.join(results_dir, f"results.csv")
    if os.path.isfile(results_file):
        rdf = pd.read_csv(results_file)
    else: 
        rdf = pd.DataFrame()
    rdf = pd.concat([rdf,mdf],ignore_index=True)
    rdf.to_csv(results_file,float_format = '%.3f',index=False)

def concatenate_results():
    # This gets all the intermediate files
    # zones = ['C1','C2','C3','N1','N2','N3','NE1','NE2','NE3','S1','S2']
    zones = ['N2','N3','NE1','NE2','NE3']
    res_dir = os.path.join(RESULTS_DIR, 'Val','intermediate-results')
    rfiles = [f for f in os.listdir(res_dir) if '.csv' in f]
    dfs = []
    for f in rfiles:
        fpath = os.path.join(res_dir,f)
        df = pd.read_csv(fpath)
        dfs.append(df)

    intermediate_files_df = pd.concat(dfs, ignore_index=True)

    # For each zone, we get the corresponding intermediate files and the full-results file, if it exists
    # from the intermediate files df, we get the evals that have been run for every year. We select
    # the evals that have been run for every year and andd them to the full-results file. After we save
    # we delete those intermediate files. 
    for zone in zones:
        zdf = intermediate_files_df.loc[intermediate_files_df.Zone == zone,:]
        evals = zdf.groupby('EvalName').size().reset_index(name='N')
        completed_evals = evals.loc[evals.N == 8,'EvalName']
        zone_fname = os.path.join(RESULTS_DIR,'Val','full-results',f"{zone}.csv")
        zrdf = pd.DataFrame()
        if Path(zone_fname).exists():
            zrdf = pd.read_csv(zone_fname)

        zdf = zdf.loc[zdf.EvalName.isin(completed_evals),:]
        zrdf = pd.concat([zrdf,zdf], ignore_index=True)

        outpath = os.path.join(RESULTS_DIR, 'Val','full-results',f"{zone}.csv")
        zrdf.to_csv(outpath, index=False, float_format='%.3f')
        zrdf['Eval_Zone'] = zrdf['EvalName'] + '_' + zrdf['Zone']

        # TODO: edit to conform to c_k change
        zone_files = [os.path.join(res_dir,f) for f in rfiles if '_'.join(f.split('_')[:-1]) in zrdf.Eval_Zone.values]
        [Path(f).unlink() for f in zone_files]

def create_baseline_results():
    res_dir = os.path.join(RESULTS_DIR,'Val','full-results')
    rfiles = [f for f in os.listdir(res_dir) if '.csv' in f]
    dfs = []
    for f in rfiles:
        fpath = os.path.join(res_dir,f)
        df = pd.read_csv(fpath)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    cols = ['DeltaCE','MaxDeltaCE', 'RIB','BetterOff','MaxBetterOff','Premium','Cost_II', 'Cost_PI','w_0','c_k','EvalName','Zone']
    indices = df.groupby(['Zone','c_k','w_0'])['DeltaCE'].idxmax()
    rdf = df.loc[indices, cols]
    outpath = os.path.join(RESULTS_DIR,'baseline_results.csv')
    rdf.to_csv(outpath,index=False, float_format='%.3f')

def summarize_results(df, cols, groupby):
    df[cols] = df[cols].to_numpy()*df['Size'].to_numpy()[:,np.newaxis]
    rdf = df.groupby(groupby)[cols + ['Size']].sum().reset_index()
    rdf[cols] = rdf[cols].to_numpy()/rdf['Size'].to_numpy()[:,np.newaxis]
    return rdf

def run_model_selection_CV(zone, method, w_0=0.1, alpha=3, c_k=0):
    results_fname = os.path.join(PREDICTIONS_DIR, f"baseline_results.csv")
    rdf = pd.read_csv(results_fname)
    models = rdf['ModelName'].unique()
    eval_set = 'Val'
    test_years = np.arange(2015,2023)
    param_dicts = []
    for model_name, year in itertools.product(models, test_years):
        pdict = create_param_dict(model_name, zone, year, eval_set, w_0, alpha, c_k)
        pdict['EvalName'] = create_eval_name(method,model_name, pdict)
        param_dicts.append(pdict)

    zone_existing_evals = check_existing_evaluations(zone, method)
    param_dicts = [p for p in param_dicts 
                    if (p['EvalName'],p['zone'],p['test_year']) not in zone_existing_evals]

    for params in tqdm(param_dicts):
        # print(params['EvalName'])
        try:
            design_contracts(method, params)
        except cp.error.SolverError:
            print(f"{params} failed to run")
            pass

def check_existing_evaluations(zone, method):
    # we'll only do this for validation set, because on the test set we'll only test the best model.
    zone_results_file = os.path.join(RESULTS_DIR,'Val','full-results',f"{zone}.csv")
    current_combos = []
    if Path(zone_results_file).exists():
        zdf = pd.read_csv(zone_results_file)
        zdf = zdf.loc[zdf.Method == method,:]
        current_combos += list(zdf['Combo'])

    intermediate_results_dir = os.path.join(RESULTS_DIR, 'Val', 'intermediate-results')
    intermediate_files = [f[:-4] for f in os.listdir(intermediate_results_dir) if zone in f]
    intermediate_files = [f for f in intermediate_files if method in f]

    for f in intermediate_files:
        zone_year = f.split('_')[-2:]
        eval_name = f.replace('_'.join(zone_year),'')[:-1]
        temp_zone, year = zone_year
        current_combos.append((eval_name, temp_zone, int(year)))

    return set(current_combos)

def create_model_param_dicts(model_name, test_years, eval_set='Val'):
    zones = ['C1','C2','C3','N1','N2','N3','NE1','NE2','NE3','S1','S2']
    base_dict = {'epsilon_p': 0.01, 'c_k': 0.13, 'subsidy': 0, 'w_0': 0.5, 'premium_ub':0.25,  
            'risk_coef': 1.5, 'S': 1, 'market_loading': 1, 'model_name': model_name, 'eval_set': eval_set}
    param_dicts = []
    for zone, test_year in itertools.product(zones, test_years):
        ndict = copy.deepcopy(base_dict)
        ndict['zone'] = zone
        ndict['test_year'] = test_year
        param_dicts.append(ndict)

def create_param_dicts(test_years, eval_set='Test'):
    results_fname = os.path.join(PREDICTIONS_DIR,f"baseline_results.csv")
    rdf = pd.read_csv(results_fname)
    models = rdf.ModelName.unique()
    regions = ['C1','C2','C3','N1','N2','N3','NE1','NE2','NE3','S1','S2']
    # regions = ['NE1','NE2','NE3']
    base_dict = {'epsilon_p': 0.01, 'c_k': 0.13, 'subsidy': 0, 'w_0': 0.5, 'premium_ub':0.25,  
            'risk_coef': 1.5, 'S': 1, 'market_loading': 1}
    param_dicts = []
    for model in models:
        for region in regions:
            for year in test_years:
                ndict = copy.deepcopy(base_dict)
                ndict['model_name'] = model
                ndict['zone'] = region
                ndict['test_year'] = year
                ndict['eval_set'] = eval_set
                param_dicts.append(ndict)
    return param_dicts

def create_param_dict(model, zone, test_year, eval_set, w_0=0.1, alpha=3, c_k=0):
    params = {'epsilon_p': 0.01, 'c_k': c_k, 'subsidy': 0, 'w_0': w_0, 'premium_ub':0.25,  
            'risk_coef': alpha, 'S': [1], 'market_loading': 1, 'zone': zone, 'eval_set': eval_set,
            'test_year': test_year, 'model_name': model}
    return params


w_0 = 0.1
c_k = 0.02
alpha = 1.5
zones = ['N2','N3','NE1','NE2','NE3']
generate_payouts(c_k, w_0, alpha, 'Chantarat')
generate_payouts(c_k, w_0, alpha, 'VMX')
# for alpha in [1.1,1.5,2,2.5,3,3.5]:
    # for zone in zones:
    #     run_model_selection_CV(zone, 'VMX', w_0, alpha, c_k)
    #     run_model_selection_CV(zone, 'Chantarat', w_0, alpha, c_k)

    # concatenate_results()
    # generate_payouts(c_k, w_0, alpha, 'Chantarat')
    # generate_payouts(c_k, w_0, alpha, 'VMX')
# create_baseline_results()
 


# ok, this is how it will be conceptually organized. This file will be for contract design and cross validation. 
# there will be a separate file that will handle the evaluation on the test set. The current way of organizing the c_k's 
# seems to be ok, the full results are all put into a single file. 

# 1. Need to write a function that will select the best models for each zone for each c_k and then generate payouts for the
# test set from these models. 
# 2. I need to implement the evaluation file
# 3. I need to implement the chantarat contract design thing too. 