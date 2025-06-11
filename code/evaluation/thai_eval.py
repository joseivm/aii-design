import pandas as pd
import os
import numpy as np
import time
import re
import copy
from pathlib import Path
import itertools
import sys
sys.path.insert(0,'/Users/joseivelarde/Projects/aii-design/code/evaluation')
from thai_synthetic_data import simulate_zone_payouts

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
PAYOUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'evaluation','Thailand','Test','payouts')

# ZONES = ['C2','C3','NE1','NE2','NE3','N2','N3','S1','S2']
ZONES = ['NE1','NE2','NE3','N2','N3']
TEST_YEARS = np.arange(2015, 2023)

##### Data loading ##### 
def load_payouts(method, c_k, w_0, alpha):
    payout_dir = os.path.join(PAYOUTS_DIR, f"{method} ck{c_k} w{w_0} r{alpha}".replace('.',''))
    payout_dfs = []
    for zone, year in itertools.product(ZONES, TEST_YEARS):
        fname = os.path.join(payout_dir,f"{zone}_{year}.csv")
        df = pd.read_csv(fname)
        payout_dfs.append(df)

    pdf = pd.concat(payout_dfs, ignore_index=True)
    pdf['Tambon'] = pdf['Idx'].apply(lambda x: x.split('-')[0])
    pdf['Year'] = pdf['Idx'].apply(lambda x: x.split('-')[1]).astype(int)
    ldf = load_loss_data()
    pdf = pdf.merge(ldf[['Idx','Weight']],on='Idx')
    if method == 'Chen':
        pdf['PredLoss'] = pdf['Payout']
        pdf['Premium'] = pdf.Premium.apply(lambda x: re.sub('[^0-9.]','',x))
        pdf['Premium'] = pdf.Premium.astype('float64')

    return pdf

def load_loss_data():
    fpath = os.path.join(PROJECT_DIR,'data','processed','Thailand_loss_data.csv')
    df = pd.read_csv(fpath)
    df.rename(columns={'ObsID':'Idx','WeightSum':'Weight'},inplace=True)
    return df

##### Evaluation #####
def create_table(c_k, w_0, alpha):
    metrics = ['Method','RIB','RIB_OG','RIB_Diff','DeltaCE','MaxDeltaCE','Premium','Cost_II','Cost_PI','CapShare']
    methods = ['VMX','Chantarat','Chen']
    rdfs = []
    for method in methods:
        rdf = get_results(method, c_k, w_0, alpha)
        rdfs.append(rdf)

    df = pd.DataFrame(rdfs)
    df = df[metrics]
    fname = f"ck{c_k}_r{alpha}_w{w_0}".replace('.','')
    df.to_csv(f"experiments/evaluation/Thailand/Test/single-zone results/{fname}_results.csv",index=False, float_format='%.3f')
    # return df[metrics]

def create_mz_table(c_k, w_0, alpha):
    metrics = ['Method','RIB','RIB_OG','RIB_Diff','DeltaCE','MaxDeltaCE','Premium','Cost_II','Cost_PI','CapShare']
    methods = ['VMX-M','VMX','Chantarat','Chen']
    rdfs = []
    for method in methods:
        rdf = get_mz_results(method, c_k, w_0, alpha)
        rdfs.append(rdf)

    df = pd.DataFrame(rdfs)
    df = df[metrics]
    fname = f"ck{c_k}_r{alpha}_w{w_0}".replace('.','')
    df.to_csv(f"experiments/evaluation/Thailand/Test/multi-zone results/{fname}_results.csv",index=False, float_format='%.3f')
    # return df[metrics]

def get_results(method, c_k, w_0, alpha=3.5):
    pdf = load_payouts(method, c_k, w_0, alpha)
    
    for zone in pdf.Zone.unique():
        zdf = pdf.loc[(pdf.Zone == zone),:]
        rdf = performance_metrics(zdf, c_k, w_0, alpha=alpha)
        worse = rdf['U_NI'] > rdf['U_PI']
        # print(f"{zone}: DeltaCE:{rdf['DeltaCE']} Max {rdf['MaxDeltaCE']} RIB: {rdf['RIB']} Worse: {worse}")

    # results = performance_metrics(pdf, c_k, w_0, alpha)
    rdf = performance_metrics(pdf, c_k, w_0, alpha)
    rdf['Method'] = method
    # print(f"Overall: DeltaCE:{rdf['DeltaCE']} Max {rdf['MaxDeltaCE']} RIB: {rdf['RIB']} ")
    return rdf

def get_mz_results(method, c_k, w_0, alpha=2):
    pdf = load_payouts(method, c_k, w_0, alpha)
    ldf = load_loss_data()
    zone_sizes = ldf.groupby(['Zone','Year'])['Weight'].sum().groupby('Zone').mean().loc[pdf.Zone.unique()]

    results = performance_metrics(pdf, c_k, w_0, alpha, premium_kwargs={'zone_sizes':zone_sizes})
    results['Method'] = method
    return results

def add_ni_metrics(df, w_0, alpha=1.5):
    df['NI_Wealth'] = w_0 + 1 - df['Loss']
    df['NI_Utility'] = 1/(1-alpha)*df['NI_Wealth']**(1-alpha)
    return df

def add_utility_metrics(df, w_0, alpha=1.5, subsidy=0):
    df['Wealth'] = w_0 - (1-subsidy)*df['Premium'] + 1 - df['Loss'] + df['Payout']
    df['Utility'] = 1/(1-alpha)*df['Wealth']**(1-alpha)

    df['NI_Wealth'] = w_0 + 1 - df['Loss']
    df['NI_Utility'] = 1/(1-alpha)*df['NI_Wealth']**(1-alpha)

    pi_premiums = df.groupby('Zone')['Loss'].mean().reset_index(name='PI_Premium')
    df = df.merge(pi_premiums, on='Zone')
    df['PI_Wealth'] = w_0 + 1 -df['PI_Premium']
    df['PI_Utility'] = 1/(1-alpha)*df['PI_Wealth']**(1-alpha)
    return df

def create_ii_df(payout_df, c_k, premium_kwargs):
    df = payout_df.copy().drop(columns=['Premium'])
    train_df = df.loc[df.Set == 'Train',:].copy()
    test_df = df.loc[df.Set == 'Test',:].copy()

    if premium_kwargs is None:
        for year in df.TestYear.unique():
            ydf = train_df.loc[train_df.TestYear == year,:]
            sim_preds = simulate_zone_payouts(ydf,payout_col='Payout',n_sim=2000, random_state=1)
            sim_preds = pd.melt(sim_preds,id_vars='Year',var_name='Zone',value_name='Payout')

            for zone in df.Zone.unique():
                zdf = ydf.loc[ydf.Zone == zone,:]
                req_capital_df = sim_preds.loc[sim_preds.Zone == zone]
                premium, capital_cost = calculate_sz_premium(zdf, c_k, req_capital_df)
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'Premium'] = premium
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'CapCost'] = capital_cost


    else:
        for year in df.TestYear.unique():
            ztrain = train_df.loc[(train_df.TestYear == year),:]
            premiums, cap_costs = calculate_mz_premiums(ztrain, c_k, **premium_kwargs)
            
            for zone in premiums.keys():
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'Premium'] = premiums[zone]
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'CapCost'] = cap_costs[zone]

    return test_df

def create_pi_df(payout_df, c_k, premium_kwargs):
    pi_df = payout_df.copy().drop(columns=['Premium'])
    pi_df['Payout'] = pi_df['Loss']
    test_df = pi_df.loc[pi_df.Set == 'Test',:].copy()

    sim_preds = simulate_zone_payouts(test_df,payout_col='Payout',n_sim=2000, random_state=1)
    sim_preds = pd.melt(sim_preds,id_vars='Year',var_name='Zone',value_name='Payout')

    if premium_kwargs is None:
        for zone, year in itertools.product(pi_df.Zone.unique(),pi_df.TestYear.unique()):
            req_capital_df = sim_preds.loc[sim_preds.Zone == zone,:]
            zdf = test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),:]
            premium, capital_cost = calculate_sz_premium(zdf, c_k, req_capital_df)
            test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'Premium'] = premium
            test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'CapCost'] = capital_cost

    else:
        for year in pi_df.TestYear.unique():
            ztest = test_df.loc[(test_df.TestYear == year),:]
            premiums, cap_costs = calculate_mz_premiums(ztest, c_k, **premium_kwargs, req_capital_df=test_df)
            
            for zone in premiums.keys():
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'Premium'] = premiums[zone]
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'CapCost'] = cap_costs[zone]
    
    return test_df

def create_ni_df(payout_df):
    ni_df = payout_df.copy()
    ni_df['Premium'] = 0
    ni_df['Payout'] = 0
    ni_df = ni_df.loc[ni_df.Set == 'Test',:]
    return ni_df

def create_ii_og_df(payout_df, c_k, premium_kwargs):
    df = payout_df.copy().drop(columns=['Premium'])
    df['Payout'] = df['PredLoss']
    train_df = df.loc[df.Set == 'Train',:].copy()
    test_df = df.loc[df.Set == 'Test',:].copy()
    
    if premium_kwargs is None:
        for year in df.TestYear.unique():
            ydf = train_df.loc[train_df.TestYear == year,:]
            sim_preds = simulate_zone_payouts(ydf,payout_col='Payout',n_sim=2000)
            sim_preds = pd.melt(sim_preds,id_vars='Year',var_name='Zone',value_name='Payout')

            for zone in df.Zone.unique():
                zdf = ydf.loc[ydf.Zone == zone,:]
                req_capital_df = sim_preds.loc[sim_preds.Zone == zone]
                premium, capital_cost = calculate_sz_premium(zdf, c_k, req_capital_df)
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'Premium'] = premium
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'CapCost'] = capital_cost

    else:
        for year in df.TestYear.unique():
            ztrain = train_df.loc[(train_df.TestYear == year),:]
            premiums, cap_costs = calculate_mz_premiums(ztrain, c_k, **premium_kwargs)
            
            for zone in premiums.keys():
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'Premium'] = premiums[zone]
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'CapCost'] = cap_costs[zone]

    return test_df

def performance_metrics(payout_df, c_k, w_0=0.1, alpha=1.5, premium_kwargs=None):
    ii_df = create_ii_df(payout_df, c_k, premium_kwargs)
    ni_df = create_ni_df(payout_df)
    pi_df = create_pi_df(payout_df, c_k, premium_kwargs)
    ii_og_df = create_ii_og_df(payout_df, c_k, premium_kwargs)

    ce_ii = certainty_equivalent(ii_df, w_0=w_0, alpha=alpha)
    ce_ii_og = certainty_equivalent(ii_og_df, w_0=w_0, alpha=alpha)
    ce_ni = certainty_equivalent(ni_df, w_0=w_0, alpha=alpha)
    ce_pi = certainty_equivalent(pi_df, w_0=w_0, alpha=alpha)
    
    delta_ce = 100*(ce_ii - ce_ni)/ce_ni
    delta_ce_og = 100*(ce_ii_og - ce_ni)/ce_ni
    max_delta_ce = 100*(ce_pi - ce_ni)/ce_ni
    rib = np.nan if max_delta_ce == 0 else delta_ce/max_delta_ce
    rib_og = np.nan if max_delta_ce == 0 else delta_ce_og/max_delta_ce
    rib_diff = 100*(rib-rib_og)/rib_og

    utility_ii = CRRA_utility(ii_df, w_0=w_0, alpha=alpha)
    utility_ii_og = CRRA_utility(ii_og_df, w_0=w_0, alpha=alpha)
    utility_ni = CRRA_utility(ni_df, w_0=w_0,alpha=alpha)
    utility_pi = CRRA_utility(pi_df, w_0=w_0, alpha=alpha)

    cap_share = (ii_df.loc[ii_df.Premium > 0,'CapCost']/ii_df.loc[ii_df.Premium > 0,'Premium']).mean()
    # better_off = pct_better_off(edf.Loss, edf.Payout, edf.Premium.mean(), w_0=w_0)
    # max_better_off = pct_better_off(edf.Loss, edf.Loss, edf.Loss.mean(), w_0=w_0)

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
        'RIB_OG':rib_og,
        'RIB_Diff': rib_diff,
        'CE_II': ce_ii,
        'CE_NI': ce_ni,
        'CE_II_OG': ce_ii_og,
        'CE_PI': ce_pi,
        'CapShare': cap_share,
        # 'BetterOff': better_off,
        # 'MaxBetterOff': max_better_off,
        'Premium': ii_df['Premium'].mean(),
        'Cost_II': ii_df['Payout'].mean(),
        'Cost_PI': pi_df['Loss'].mean(),
        'Size' : len(ii_df),
        'w_0' : w_0
    }
    metrics_dict = {key: np.round(value,3) for key, value in metrics_dict.items()}
    return metrics_dict

def certainty_equivalent(edf,w_0=0.5, alpha=1.5, markup=0):
    edf['Wealth'] = w_0 - (1+markup)*edf['Premium'] + 1 - edf['Loss'] + edf['Payout']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    edf['WUtility'] = edf['Utility']*edf['Weight']
    # average_utility = edf['Utility'].mean()
    average_utility = edf['WUtility'].sum()/edf['Weight'].sum()
    certainty_equivalent = ((1-alpha)*(average_utility))**(1/(1-alpha))
    return certainty_equivalent

def CRRA_utility(edf, w_0=0.5, alpha=1.5, markup=0):
    edf['Wealth'] = w_0 - (1+markup)*edf['Premium'] + 1 - edf['Loss'] + edf['Payout']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    edf['WUtility'] = edf['Utility']*edf['Weight']
    return edf['WUtility'].sum()/edf['Weight'].sum()
    # return edf['Utility'].mean()

def pct_better_off(y_true, y_pred, premium, w_0=0.5, alpha=1.5, markup=0):
    edf = pd.DataFrame({'Loss':y_true, 'Payout':y_pred})
    edf['Wealth'] = w_0 - (1+markup)*premium + 1 - edf['Loss'] + edf['Payout']
    edf['Wealth_NI'] = w_0 + 1 - edf['Loss']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    edf['Utility_NI'] = 1/(1-alpha)*edf['Wealth_NI']**(1-alpha)
    edf['BetterOff'] = edf['Utility'] > edf['Utility_NI']
    return edf['BetterOff'].mean()

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

def calculate_mz_premiums(df, c_k, zone_sizes, req_capital_df=None):

    cap_df = (req_capital_df if req_capital_df is not None else df)
        
    sim_preds = simulate_zone_payouts(cap_df,payout_col='Payout',n_sim=2000, random_state=1)
    sim_preds = pd.melt(sim_preds,id_vars='Year',var_name='Zone',value_name='Payout')
    payouts = sim_preds.merge(zone_sizes,on='Zone')
    payouts['TotalPayout'] = payouts['Payout']*payouts['Weight']
    annual_totals = payouts.groupby('Year')['TotalPayout'].sum().reset_index()

    payout_cvar = CVaR(annual_totals, 'TotalPayout', 'TotalPayout', 0.01)
    average_payout = annual_totals['TotalPayout'].mean()
    required_capital = payout_cvar - average_payout

    # 5) total capital charge in dollars
    C_tot = c_k * required_capital

    # 6) each zone’s average per‐unit payout
    avg_share = df.groupby('Zone')['Payout'].mean()

    # 7) portfolio average dollar loss
    #    = sum_z S_z * avg_share[z]
    L_avg = (zone_sizes * avg_share).sum()


    # 8) allocate capital per unit by zone’s loss‐share
    #    cost_per_unit[z] = C_tot * avg_share[z] / L_avg
    cap_costs = (C_tot * avg_share) / L_avg
    cap_costs.fillna(0,inplace=True)

    # 9) build final premiums
    premiums = {}
    for z in zone_sizes.keys():
        # zones with zero avg_share get zero total premium
        if avg_share.get(z, 0.0) > 0:
            premiums[z] = avg_share[z] + cap_costs[z]
        else:
            premiums[z] = 0.0
         
    return premiums, cap_costs

def calculate_mz_premiums_old(df, c_k, zone_sizes):
    # TODO: fix this, make it similar to the one in thai_mz_contract_design. We dont have
    # a sim_df here, so consider making it using df. 
    years = df.TestYear.unique()
    df['WeightedPayout'] = df['Payout']*df['Weight']
    total_payouts = []
    for year in years: 
        total_payout = df.loc[df.TestYear == year,'WeightedPayout'].sum()
        total_payouts.append({'Year':year,'TotalPayout':total_payout})

    pdf = pd.DataFrame(total_payouts)

    payout_cvar = CVaR(pdf, 'TotalPayout', 'TotalPayout', 0.01)
    average_payout = pdf['TotalPayout'].mean()
    required_capital = payout_cvar - average_payout

    premiums = {}
    for zone in df.Zone.unique():
        premiums[zone] = df.loc[df.Zone == zone,'Payout'].mean() + c_k*required_capital/np.sum(zone_sizes)
         
    return premiums

def CVaR(df,loss_col,outcome_col,epsilon=0.2,weight_col=None):
    q = np.quantile(df[loss_col],1-epsilon)
    if weight_col is None:
        cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    else:
        cvar = df.loc[df[loss_col] >= q,outcome_col].sum()/df.loc[df[loss_col] >= q,weight_col].sum()
    return cvar

def debugging():
    # TODO: re generate mz payouts, modify this so that it uses weighted utility

    w_0 = 0.1
    c_k = 0.02
    alpha = 1.5
    df = load_payouts('VMX-M',c_k,w_0,alpha)
    df = add_utility_metrics(df, w_0, alpha)

    df['UtilityDiff'] = df['Utility'] - df['NI_Utility']
    df['PosLoss'] = df.Loss > 0
    df['UtilityGain'] = df.UtilityDiff > 0

    df.groupby(['PosLoss','UtilityGain'])['UtilityDiff'].sum()



w_0 = 0.1
method = 'VMX'
alpha = 1.5
c_k = 0.02

# df = create_table(c_k,w_0,alpha)
# fname = f"ck{c_k}_r{alpha}_w{w_0}".replace('.','')
# df.to_latex(f"experiments/evaluation/Thailand/Test/{fname}_results.tex",index=False, float_format='%.3f')
# for alpha in [1.1, 1.5, 2, 2.5, 3, 3.5]:
    # print(f"Alpha: {alpha}")
    # get_results('VMX',c_k,w_0,alpha)
    # for c_k in [0.02]:
    #     print(f"Alpha {alpha} C_k {c_k}")
    #     create_table(c_k,w_0,alpha)
    #     create_mz_table(c_k, w_0, alpha)
        # fname = f"ck{c_k}_r{alpha}_w{w_0}".replace('.','')
        # df.to_csv(f"experiments/evaluation/Thailand/Test/{fname}_results.csv",index=False, float_format='%.3f')


create_table(c_k,w_0,alpha)
create_mz_table(c_k, w_0, alpha)