import pandas as pd
import os
import numpy as np
import time
import sys
sys.path.insert(0,'/Users/joseivelarde/Projects/aii-design/code/evaluation')
import cvxpy as cp
import math
from pathlib import Path
import itertools
from tqdm import tqdm
from thai_synthetic_data import simulate_zone_payouts

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

# TODO: I need to think about when I'm using the test set and when I want to use the training set. 
# I also need to make sure my use of "TestYear" is consistent. 

##### Data Loading #####
def load_all_zone_preds(zones, params):
    c_k, w_0, alpha = params['c_k'], params['w_0'], params['risk_coef']
    zdfs = []
    for zone in zones:
        zdf = load_zone_preds(zone, c_k, w_0, alpha)
        zdfs.append(zdf)

    df = pd.concat(zdfs, ignore_index=True)
    df.loc[df.PredLoss > 1, 'PredLoss'] = 1
    df.loc[df.PredLoss < 0, 'PredLoss'] = 0
    df['Year'] = df.Idx.apply(lambda x: x.split('-')[1]).astype(int)
    return df

def load_zone_preds(zone, c_k, w_0, alpha, method='VMX'):
    model_name = get_best_model(zone, c_k, w_0, alpha, method)
    dfs = []
    for year in TEST_YEARS:
        df = load_model_predictions(model_name, zone, year)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df['Province'] = df.Idx.apply(lambda x: x[1:3])
    return df

def load_model_predictions(model_name, zone, test_year):
    pred_dir = os.path.join(PREDICTIONS_DIR,'model-preds',model_name)
    pred_file = os.path.join(pred_dir,f"{zone}_{test_year}_preds.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

def load_loss_data():
    fpath = os.path.join(DATA_DIR,'processed','Thailand_loss_data.csv')
    df = pd.read_csv(fpath)
    df['Idx'] = df['ObsID']
    return df

def create_multi_zone_data_old(zones, test_year, params, stratify=True, q=10, sample=550):
    # TODO: consider just using copulas
    # TODO: add weights
    ldf = load_loss_data()
    c_k, w_0, alpha = params['c_k'], params['w_0'], params['risk_coef']
    train_dfs = {}
    for zone in zones: 
        zdf = load_zone_preds(zone, c_k, w_0, alpha)
        zdf = zdf.loc[(zdf.TestYear != test_year) & (zdf.Set == 'Test'),:]
        # zdf = zdf.merge(ldf[['Idx','WeightSum']],on='Idx')
        # zdf['WeightedLoss'] = zdf['Loss']*zdf['WeightSum']
        # zdf['WeightedPredLoss'] = zdf['PredLoss']*zdf['WeightSum']
        # zdf = zdf.groupby(['TestYear','Zone','Province'])[
        #     ['WeightedLoss','WeightedPredLoss','WeightSum']].sum().reset_index()
        # zdf['Loss'] = zdf['WeightedLoss']/zdf['WeightSum']
        # zdf['PredLoss'] = zdf['WeightedPredLoss']/zdf['WeightSum']
        train_dfs[zone] = zdf

    yearly_sample = sample
    train_years = [yr for yr in np.arange(2015,2023) if yr != test_year]
    all_year_dfs = []
    for year in train_years:
        year_dfs = []
        for zone in zones:
            zdf = train_dfs[zone]
            if stratify:
                zdf = zdf.loc[zdf.TestYear == year, ['Loss','PredLoss']]
                zdf['stratum'] = pd.qcut(zdf['Loss'],q=q, duplicates='drop')
                if zdf.stratum.nunique() == 0:
                    zdf['stratum'] = 'None'
                zdf = zdf.groupby('stratum',dropna=False,observed=True).sample(n=(yearly_sample+10)//zdf.stratum.nunique(),replace=True)
            else:
                zdf = zdf.loc[zdf.TestYear == year, ['Loss','PredLoss']].sample(yearly_sample,replace=True, random_state=5)
            zdf.rename(columns={'Loss':f"Loss_{zone}",'PredLoss':f"PredLoss_{zone}"}, inplace=True)
            zdf.reset_index(inplace=True, drop=True)
            zdf.drop(columns=['stratum'],inplace=True,errors='ignore')
            year_dfs.append(zdf.head(yearly_sample))

        ydf = pd.concat(year_dfs, axis=1)
        ydf['Year'] = year
        all_year_dfs.append(ydf)

    train_df = pd.concat(all_year_dfs, ignore_index=True)
    return train_df

def create_training_sample(pdf, sample=2000):
    ldf = load_loss_data()
    ldf = ldf.loc[ldf.Zone.isin(pdf.Zone.unique()),:]
    pdf = pdf.copy()
    pdf['stratum'] = np.where(pdf.Loss == 0, 'zero','positive')

    zone_exposures = ldf.groupby(['Zone','Year'])['WeightSum'].sum().groupby('Zone').mean()
    zone_factor = (zone_exposures/zone_exposures.sum()).to_dict()

    pop_zs = pdf.groupby(['Zone','stratum']).size().to_dict()
    pop_z = pdf['Zone'].value_counts().to_dict()

    N = sample
    n_zs = {}
    for (z,s), cnt_zs in pop_zs.items():
        q_zs = cnt_zs / pop_z[z]
        n_zs[(z,s)] = math.ceil(N * zone_factor[z] * q_zs)

    samples = []
    for (z,s), n_draw in n_zs.items():
        grp = pdf[(pdf['Zone']==z) & (pdf['stratum']==s)]
        # if n_draw > len(grp) you could fallback to replace=True
        replace = n_draw > len(grp)
        samples.append(grp.sample(n=n_draw, replace=replace, random_state=42))
    sample_df = pd.concat(samples).reset_index(drop=True)

    pop_idx = pd.MultiIndex.from_tuples(pop_zs.keys(), names=['Zone','stratum'])
    pop_s   = pd.Series(pop_zs.values(), index=pop_idx, name='pop_zs')

    n_idx  = pd.MultiIndex.from_tuples(n_zs.keys(), names=['Zone','stratum'])
    n_s    = pd.Series(n_zs.values(), index=n_idx, name='n_zs')

    # 3) Now join them onto your sample_df in one go:
    sample_df = (
        sample_df
        .join(pop_s, on=['Zone','stratum'])
        .join(n_s,   on=['Zone','stratum'])
    )

    # 4) Finally compute the weights vectorized:
    sample_df['w'] = sample_df['pop_zs'] / sample_df['n_zs']

    return sample_df

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
    K_z       = cp.Variable(Z)

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

    constraints += [
        cp.sum(K_z) == K,
        K_z >= 0
    ]

    # premium definition & caps
    constraints += [
        pi == (1/n_sim)*cp.sum(alpha_sim, axis=0)
      + cp.multiply(c_k / zone_sizes, K_z),
        max_premium >= market_loading*pi,
        b >= 0,
        (1/n_sim)*cp.sum(alpha_sim, axis=0) >= cp.multiply(c_k / zone_sizes, K_z) - 1e-8
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
    obj = prob.solve(max_iter=1000)

    return a.value, b.value, K_z.value

def add_opt_payouts(df, a, b, agg_level='Zone'):
    zdfs = []
    for zone in a.keys():
        zdf  = df.loc[df[agg_level] == zone, :].copy()
        zdf['Payout'] = np.minimum(a[zone]*zdf['PredLoss'] - b[zone], 1)
        zdf['Payout'] = np.maximum(0, zdf['Payout'])
        zdfs.append(zdf)

    return pd.concat(zdfs, ignore_index=True)

def calculate_premiums(df, sim_df, c_k, zone_sizes):
    payouts = sim_df.groupby(['Zone','Year'])['Payout'].mean().reset_index()
    payouts = payouts.merge(zone_sizes,on='Zone')
    payouts['TotalPayout'] = payouts['Payout']*payouts['WeightSum']
    annual_totals = payouts.groupby('Year')['TotalPayout'].sum().reset_index()

    payout_cvar = CVaR(annual_totals, 'TotalPayout', 'TotalPayout', 0.01)
    average_payout = annual_totals['TotalPayout'].mean()
    required_capital = payout_cvar - average_payout
    per_unit_cost_of_capital = c_k*required_capital/zone_sizes.sum()

    premiums = {}
    for zone in df.Zone.unique():
        premiums[zone] = df.loc[df.Zone == zone,'Payout'].mean() + per_unit_cost_of_capital
         
    return premiums, per_unit_cost_of_capital

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

##### MZ Eval #####
def design_mz_contracts(params):
    # ZONES = ['NE1','NE2','NE3','N2','N3']
    # params = create_param_dict(ZONES,0.07,2017,alpha=3)
    zones, test_year = params['zones'], params['test_year']
    zones = sorted(zones)

    # Load all predictions
    model_preds = load_all_zone_preds(zones, params)
    model_preds = model_preds.loc[model_preds.TestYear == test_year,:]
    train_df = model_preds.loc[model_preds.Set == 'Val',:]
    test_df = model_preds.loc[model_preds.Set == 'Test',:]
    mz_train_df = create_training_sample(train_df, sample=2000)

    # Get training preds
    sim_preds = simulate_zone_payouts(train_df,n_sim=2000)
    sim_matrix = sim_preds.loc[:,zones].to_numpy()

    # Get zone sizes
    ldf = load_loss_data()
    sizes = ldf.groupby(['Zone','Year'])['WeightSum'].sum().groupby('Zone').mean().loc[zones]
    params['S'] = sizes

    # Design contracts
    # start = time.time()
    a, b, K_z = optimization_program(sim_matrix, mz_train_df, params)
    # end = time.time()
    # print(f"Runtime: {(end-start)/60}")
    a = {zone: val for zone, val in zip(zones, a)}
    b = {zone: val for zone, val in zip(zones, b)}
    Ks = {zone: val/sum(K_z) for zone, val in zip(zones,K_z)}

    # Get test_preds
    sim_preds = pd.melt(sim_preds,id_vars='Year',var_name='Zone',value_name='PredLoss')
    opt_train_payouts = add_opt_payouts(train_df, a, b)
    sim_train_payouts = add_opt_payouts(sim_preds,a,b)

    opt_premiums, req_capital = calculate_premiums(opt_train_payouts, sim_train_payouts,params['c_k'],params['S'])

    # This can also take a list/dict of dfs
    opt_test_payouts = add_opt_payouts(test_df, a, b)
    opt_eval_df = create_eval_df(opt_test_payouts, opt_premiums, params)
    opt_train_df = create_eval_df(opt_train_payouts, opt_premiums, params)

    # Save results to file
    opt_eval_df['Set'] = 'Test'
    opt_train_df['Set'] = 'Train'
    opt_eval_df = pd.concat([opt_train_df,opt_eval_df],ignore_index=True)
    payout_dir = os.path.join(RESULTS_DIR, 'Test', 'payouts', f"VMX-M ck{params['c_k']} w{params['w_0']} r{params['risk_coef']}".replace('.',''))
    Path(payout_dir).mkdir(exist_ok=True,parents=True)
    for zone in zones:
        eval_df_filename = os.path.join(payout_dir, f"{zone}_{test_year}.csv")
        zdf = opt_eval_df.loc[opt_eval_df.Zone == zone,:]
        zdf.to_csv(eval_df_filename,index=False,float_format = '%.3f')

    contract_params = {}
    for zone in zones:
        contract_params[f"{zone}_a"] = np.round(a[zone],2)
        contract_params[f"{zone}_b"] = np.round(b[zone],2)
        contract_params[f"{zone}_Kz"] = np.round(Ks[zone],3)

    # Save results to file
    param_df = pd.DataFrame([contract_params])
    pdf_fname = os.path.join(payout_dir,f"contract_params_{test_year}.csv")
    param_df.to_csv(pdf_fname,index=False)

def create_eval_df(payout_df, premiums, params):
    edfs = []
    w_0, alpha = params['w_0'], params['risk_coef']
    for zone in premiums.keys():
        edf = payout_df.loc[payout_df.Zone == zone,:].copy()
        edf['Premium'] = premiums[zone]
        edf['Wealth'] = w_0 + 1 - edf['Loss'] + edf['Payout'] - edf['Premium']
        edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
        edfs.append(edf)

    return pd.concat(edfs,ignore_index=True)

##### Generate Payouts #####
def generate_payouts(c_k, w_0, alpha):
    zones = ['N2','N3','NE1','NE2','NE3']
    test_years = np.arange(2015,2023)
    param_dicts = []
    for year in test_years:
        param_dicts.append(create_param_dict(zones, c_k, year, w_0, alpha))

    for params in tqdm(param_dicts):
        design_mz_contracts(params)

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

def create_param_dict(zones, c_k, test_year, w_0=0.1, alpha=2):
    params = {'epsilon_p': 0.01, 'c_k': c_k, 'subsidy': 0, 'w_0': w_0, 'premium_ub':0.25,  
            'risk_coef': alpha, 'market_loading': 1, 'eval_set': 'Test',
            'test_year': test_year, 'zones': zones}
    return params


def testing():
    # ZONES = ['NE1','NE2','NE3','N2','N3']
    ZONES = ['C1','C2','C3','N1','N2','N3','NE1','NE2','NE3','S1','S2']
    params = create_param_dict(ZONES,0.13,2017,alpha=1.5)


# TODO: Add the if name == 'main' thing
c_k = 0.02
w_0 = 0.1
alpha = 1.5
generate_payouts(c_k, w_0, alpha)