import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
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
    pred_method = 'VMX' if method == 'RawPreds' else method
    payout_dir = os.path.join(PAYOUTS_DIR, f"{pred_method} ck{c_k} w{w_0} r{alpha}".replace('.',''))
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
        pdf = pdf.round(3)

    pdf.loc[pdf.PredLoss < 0, 'PredLoss'] = 0
    pdf.loc[pdf.PredLoss > 1, 'PredLoss'] = 1

    if method == 'RawPreds':
        pdf['Payout'] = pdf['PredLoss']

    return pdf

def load_required_capital_shares(year, c_k, w_0, alpha):
    payout_dir = os.path.join(PAYOUTS_DIR, f"VMX-M ck{c_k} w{w_0} r{alpha}".replace('.',''))
    param_fpath = os.path.join(payout_dir,f"contract_params_{year}.csv")
    pdf = pd.read_csv(param_fpath)
    cols = [col for col in pdf.columns if 'Kz' in col]
    pdf = pdf.loc[:,cols]
    new_names = {col: col.split('_')[0] for col in cols}
    pdf.rename(columns=new_names,inplace=True)
    return pdf.squeeze(axis=0)

def load_loss_data():
    fpath = os.path.join(PROJECT_DIR,'data','processed','Thailand_loss_data.csv')
    df = pd.read_csv(fpath)
    df.rename(columns={'ObsID':'Idx','WeightSum':'Weight'},inplace=True)
    return df

##### Evaluation #####
def create_table(c_k, w_0, alpha):
    metrics = ['Method','RIB','PayoutPrecision','LossRecall','LossUtilityGain','LossUtilityLoss','DeltaCE','MaxDeltaCE','Premium','Premium_PI','Cost_II','Cost_PI','CapShare']
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
    metrics = ['Method','RIB','PayoutPrecision','LossRecall','LossUtilityGain','LossUtilityLoss','DeltaCE','MaxDeltaCE','Premium','Premium_PI','Cost_II','Cost_PI','CapShare']
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

    rdf = performance_metrics(pdf, c_k, w_0, alpha)
    rdf['Method'] = method
    # print(f"Overall: DeltaCE:{rdf['DeltaCE']} Max {rdf['MaxDeltaCE']} RIB: {rdf['RIB']} ")
    return rdf

def get_mz_results(method, c_k, w_0, alpha=2):
    # if method is VMX i need to get the required capital shares and add them to premium_kwargs
    pdf = load_payouts(method, c_k, w_0, alpha)
    ldf = load_loss_data()
    zone_sizes = ldf.groupby(['Zone','Year'])['Weight'].sum().groupby('Zone').mean().loc[pdf.Zone.unique()]
    premium_kwargs = {'zone_sizes':zone_sizes}

    if method == 'VMX-M':
        premium_kwargs['cap_shares'] = True

    results = performance_metrics(pdf, c_k, w_0, alpha, premium_kwargs)
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

    # pi_premiums = df.groupby('Zone')['Loss'].mean().reset_index(name='PI_Premium')
    # df = df.merge(pi_premiums, on='Zone')
    # df['PI_Wealth'] = w_0 + 1 -df['PI_Premium']
    # df['PI_Utility'] = 1/(1-alpha)*df['PI_Wealth']**(1-alpha)
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
            sim_preds.loc[sim_preds.Payout > 1, 'Payout'] = 1

            for zone in df.Zone.unique():
                zdf = ydf.loc[ydf.Zone == zone,:]
                req_capital_df = sim_preds.loc[sim_preds.Zone == zone]
                premium, capital_cost = calculate_sz_premium(zdf, c_k, req_capital_df)
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'Premium'] = premium
                test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'CapCost'] = capital_cost


    else:
        for year in df.TestYear.unique():
            if premium_kwargs.get('cap_shares') is not None:
                cap_shares = load_required_capital_shares(year, c_k, w_0, alpha)
                premium_kwargs['cap_shares'] = cap_shares
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
    test_df['w'] = 1

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

def performance_metrics(payout_df, c_k, w_0=0.1, alpha=1.5, premium_kwargs=None):
    ii_df = create_ii_df(payout_df, c_k, premium_kwargs)
    if premium_kwargs is not None:
        premium_kwargs['cap_shares'] = None
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

    ii_df = add_utility_metrics(ii_df, w_0, alpha)
    ii_df['UtilityDiff'] = ii_df['Utility'] - ii_df['NI_Utility']
    ii_df['PosLoss'] = ii_df.Loss > 0
    ii_df['UtilityGain'] = ii_df.UtilityDiff > 0
    ii_df['WUtilityDiff'] = ii_df['Weight']*ii_df['UtilityDiff']
    gdf = ii_df.groupby(['PosLoss','UtilityGain'])['WUtilityDiff'].sum()
    utility_gain_shares = gdf/gdf.groupby('UtilityGain').sum()

    ii_df['ValidPayout'] = np.minimum(ii_df['Loss'], ii_df['Payout'])
    ii_df['WValidPayout'] = ii_df['Weight']*ii_df['ValidPayout']
    ii_df['WPayout'] = ii_df['Weight']*ii_df['Payout']
    ii_df['WLoss'] = ii_df['Weight']*ii_df['Loss']

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
        'PayoutPrecision': ii_df['WValidPayout'].sum()/ii_df['WPayout'].sum(),
        'LossRecall': ii_df['WValidPayout'].sum()/ii_df['WLoss'].sum(),
        'AveragePayout': ii_df['Payout'].mean(),
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
        'CapShare_PI': (pi_df.loc[pi_df.Premium > 0,'CapCost']/pi_df.loc[pi_df.Premium > 0,'Premium']).mean(),
        'LossUtilityGain': utility_gain_shares.get((True,True),0),
        'LossUtilityLoss': utility_gain_shares.get((True,False),0),
        # 'BetterOff': better_off,
        # 'MaxBetterOff': max_better_off,
        'Premium': ii_df['Premium'].mean(), # should this be weighted?
        'Premium_PI': pi_df['Premium'].mean(),
        'Cost_II': ii_df['Payout'].mean(),
        'Cost_PI': pi_df['Loss'].mean(),
        'Size' : len(ii_df),
        'w_0' : w_0
    }
    metrics_dict = {key: np.round(value,3) for key, value in metrics_dict.items()}
    return metrics_dict

def performance_metrics_by_zone(payout_df, c_k, w_0=0.1, alpha=1.5, premium_kwargs=None):
    overall_ii_df = create_ii_df(payout_df, c_k, premium_kwargs)
    if premium_kwargs is not None:
        premium_kwargs['cap_shares'] = None
    overall_ni_df = create_ni_df(payout_df)
    overall_pi_df = create_pi_df(payout_df, c_k, premium_kwargs)

    results_data = []
    for zone in overall_ii_df.Zone.unique():
        ii_df = overall_ii_df.loc[overall_ii_df.Zone == zone,:].copy()
        ni_df = overall_ni_df.loc[overall_ni_df.Zone == zone,:].copy()
        pi_df = overall_pi_df.loc[overall_pi_df.Zone == zone,:].copy()

        ce_ii = certainty_equivalent(ii_df, w_0=w_0, alpha=alpha)
        ce_ni = certainty_equivalent(ni_df, w_0=w_0, alpha=alpha)
        ce_pi = certainty_equivalent(pi_df, w_0=w_0, alpha=alpha)

        delta_ce = 100*(ce_ii - ce_ni)/ce_ni
        max_delta_ce = 100*(ce_pi - ce_ni)/ce_ni
        rib = np.nan if max_delta_ce == 0 else delta_ce/max_delta_ce

        ii_df = add_utility_metrics(ii_df, w_0, alpha)
        ii_df['UtilityDiff'] = ii_df['Utility'] - ii_df['NI_Utility']
        ii_df['PosLoss'] = ii_df.Loss > 0
        ii_df['UtilityGain'] = ii_df.UtilityDiff > 0
        ii_df['WUtilityDiff'] = ii_df['Weight']*ii_df['UtilityDiff']
        gdf = ii_df.groupby(['PosLoss','UtilityGain'])['WUtilityDiff'].sum()
        utility_gain_shares = gdf/gdf.groupby('UtilityGain').sum()

        ii_df['ValidPayout'] = np.minimum(ii_df['Loss'], ii_df['Payout'])
        ii_df['WValidPayout'] = ii_df['Weight']*ii_df['ValidPayout']
        ii_df['WPayout'] = ii_df['Weight']*ii_df['Payout']
        ii_df['WLoss'] = ii_df['Weight']*ii_df['Loss']

        cap_share = (ii_df.loc[ii_df.Premium > 0,'CapCost']/ii_df.loc[ii_df.Premium > 0,'Premium']).mean()

        metrics_dict = {
        'PayoutPrecision': ii_df['WValidPayout'].sum()/ii_df['WPayout'].sum(),
        'LossRecall': ii_df['WValidPayout'].sum()/ii_df['WLoss'].sum(),
        'AveragePayout': ii_df['Payout'].mean(),
        'DeltaCE': delta_ce,
        'MaxDeltaCE': max_delta_ce,
        'RIB': rib,
        'CE_II': ce_ii,
        'CE_NI': ce_ni,
        'CE_PI': ce_pi,
        'CapShare': cap_share,
        'CapShare_PI': (pi_df.loc[pi_df.Premium > 0,'CapCost']/pi_df.loc[pi_df.Premium > 0,'Premium']).mean(),
        'LossUtilityGain': utility_gain_shares.get((True,True),0),
        'LossUtilityLoss': utility_gain_shares.get((True,False),0),
        # 'BetterOff': better_off,
        # 'MaxBetterOff': max_better_off,
        'Premium': ii_df['Premium'].mean(), # should this be weighted?
        'Premium_PI': pi_df['Premium'].mean(),
        'Cost_II': ii_df['Payout'].mean(),
        'Cost_PI': pi_df['Loss'].mean(),
        'Size' : len(ii_df),
        'w_0' : w_0,
        'Zone': zone
    }
        results_data.append(metrics_dict)

    rdf = pd.DataFrame(results_data)
    return rdf

def performance_metrics_by_year(payout_df, c_k, w_0=0.1, alpha=1.5, premium_kwargs=None):
    overall_ii_df = create_ii_df(payout_df, c_k, premium_kwargs)
    if premium_kwargs is not None:
        premium_kwargs['cap_shares'] = None
    overall_ni_df = create_ni_df(payout_df)
    overall_pi_df = create_pi_df(payout_df, c_k, premium_kwargs)

    results_data = []
    for year in overall_ii_df.TestYear.unique():
        ii_df = overall_ii_df.loc[overall_ii_df.TestYear == year,:].copy()
        ni_df = overall_ni_df.loc[overall_ni_df.TestYear == year,:].copy()
        pi_df = overall_pi_df.loc[overall_pi_df.TestYear == year,:].copy()

        ce_ii = certainty_equivalent(ii_df, w_0=w_0, alpha=alpha)
        ce_ni = certainty_equivalent(ni_df, w_0=w_0, alpha=alpha)
        ce_pi = certainty_equivalent(pi_df, w_0=w_0, alpha=alpha)

        delta_ce = 100*(ce_ii - ce_ni)/ce_ni
        max_delta_ce = 100*(ce_pi - ce_ni)/ce_ni
        rib = np.nan if max_delta_ce == 0 else delta_ce/max_delta_ce

        ii_df = add_utility_metrics(ii_df, w_0, alpha)
        ii_df['UtilityDiff'] = ii_df['Utility'] - ii_df['NI_Utility']
        ii_df['PosLoss'] = ii_df.Loss > 0
        ii_df['UtilityGain'] = ii_df.UtilityDiff > 0
        ii_df['WUtilityDiff'] = ii_df['Weight']*ii_df['UtilityDiff']
        gdf = ii_df.groupby(['PosLoss','UtilityGain'])['WUtilityDiff'].sum()
        utility_gain_shares = gdf/gdf.groupby('UtilityGain').sum()

        ii_df['ValidPayout'] = np.minimum(ii_df['Loss'], ii_df['Payout'])
        ii_df['WValidPayout'] = ii_df['Weight']*ii_df['ValidPayout']
        ii_df['WPayout'] = ii_df['Weight']*ii_df['Payout']
        ii_df['WLoss'] = ii_df['Weight']*ii_df['Loss']

        cap_share = (ii_df.loc[ii_df.Premium > 0,'CapCost']/ii_df.loc[ii_df.Premium > 0,'Premium']).mean()

        metrics_dict = {
        'PayoutPrecision': ii_df['WValidPayout'].sum()/ii_df['WPayout'].sum(),
        'LossRecall': ii_df['WValidPayout'].sum()/ii_df['WLoss'].sum(),
        'AveragePayout': ii_df['Payout'].mean(),
        'DeltaCE': delta_ce,
        'MaxDeltaCE': max_delta_ce,
        'RIB': rib,
        'CE_II': ce_ii,
        'CE_NI': ce_ni,
        'CE_PI': ce_pi,
        'CapShare': cap_share,
        'CapShare_PI': (pi_df.loc[pi_df.Premium > 0,'CapCost']/pi_df.loc[pi_df.Premium > 0,'Premium']).mean(),
        'LossUtilityGain': utility_gain_shares.get((True,True),0),
        'LossUtilityLoss': utility_gain_shares.get((True,False),0),
        # 'BetterOff': better_off,
        # 'MaxBetterOff': max_better_off,
        'Premium': ii_df['Premium'].mean(), # should this be weighted?
        'Premium_PI': pi_df['Premium'].mean(),
        'Cost_II': ii_df['Payout'].mean(),
        'Cost_PI': pi_df['Loss'].mean(),
        'Size' : len(ii_df),
        'w_0' : w_0,
        'TestYear': year
    }
        results_data.append(metrics_dict)

    rdf = pd.DataFrame(results_data)
    return rdf

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
    payout_cvar = CVaR(req_capital_df, loss_col='Payout', outcome_col='Payout', epsilon=0.01)
    average_payout = req_capital_df['Payout'].mean()
    required_capital = payout_cvar-average_payout
    cost_of_capital = c_k*required_capital
    premium = payout_df.Payout.mean() + cost_of_capital
    return (1-subsidy)*premium, cost_of_capital

def calculate_mz_premiums(df, c_k, zone_sizes, req_capital_df=None, cap_shares=None):
    df = df.copy()
    cap_df = req_capital_df if req_capital_df is not None else df
    sim_preds = simulate_zone_payouts(cap_df, payout_col='Payout', n_sim=2000, random_state=1)
    sim_preds = pd.melt(sim_preds, id_vars='Year', var_name='Zone', value_name='Payout')
    sim_preds.loc[sim_preds.Payout > 1, 'Payout'] = 1

    # Weight by zone size
    payouts = sim_preds.merge(zone_sizes, on='Zone')
    payouts['TotalPayout'] = payouts['Payout'] * payouts['Weight']
    annual_totals = payouts.groupby('Year')['TotalPayout'].sum().reset_index()

    # Capital requirement
    payout_cvar = CVaR(annual_totals, 'TotalPayout', 'TotalPayout', 0.01)
    avg_total_payout = np.average(annual_totals['TotalPayout'])
    required_capital = payout_cvar - avg_total_payout
    C_tot = c_k * required_capital

    # 6) each zone’s average per‐unit payout
    # I need to change this to be by weighted payout
    df['WPayout'] = df['Weight']*df['Payout']
    avg_share = df.groupby('Zone')['WPayout'].sum()/df.WPayout.sum()
    avg_payout = df.groupby('Zone')['Payout'].mean()

    # 7) portfolio average dollar payout
    #    = sum_z S_z * avg_share[z]
    # L_avg = (zone_sizes * avg_share).sum()


    # 8) allocate capital per unit by zone’s loss‐share
    #    cost_per_unit[z] = C_tot * avg_share[z] / L_avg
    if cap_shares is None:
        cap_costs = (C_tot * avg_share) / zone_sizes
        cap_costs.fillna(0,inplace=True)
    
    else:
        cap_costs = (C_tot*cap_shares)/zone_sizes
        cap_costs.fillna(0,inplace=True)

    # 9) build final premiums
    premiums = {
        z: avg_payout[z] + cap_costs[z] if avg_share[z] > 0 else 0.0
        for z in zone_sizes.index
    }
         
    return premiums, cap_costs

def CVaR(df, loss_col, outcome_col, epsilon=0.01, weight_col=None):
    df = df.copy()
    q = np.quantile(df[loss_col], 1 - epsilon)
    tail_df = df[df[loss_col] >= q]

    if weight_col is None:
        return tail_df[outcome_col].mean()
    else:
        return np.average(tail_df[outcome_col], weights=tail_df[weight_col])

def debugging():


    w_0 = 0.1
    c_k = 0.02
    alpha = 1.5
    df = load_payouts('VMX',c_k,w_0,alpha)
    df.loc[df.PredLoss < 0, 'PredLoss'] = 0
    df['Payout'] = df['PredLoss']
    df = df.loc[df.Set == 'Test',:]
    df = add_utility_metrics(df, w_0, alpha)

    df['UtilityDiff'] = df['Utility'] - df['NI_Utility']
    df['PosLoss'] = df.Loss > 0
    df['UtilityGain'] = df.UtilityDiff > 0
    df['WUtilityDiff'] = df['Weight']*df['UtilityDiff']

    gdf = df.groupby(['PosLoss','UtilityGain'])['WUtilityDiff'].sum()
    gdf/gdf.groupby('UtilityGain').sum()

def kde_plot_general(df, contracts=['VMX','Chen','Chantarat']):
    utilities = []
    for contract in contracts:
        u_contract = df[f"Utility_{contract}"].dropna().to_numpy()
        utilities.append(u_contract)

    kdes = []
    for utility in utilities:
        kdes.append(gaussian_kde(utility))

    xmin = -3
    xmax = max(utilities[0].max(),utilities[1].max())
    x_grid = np.linspace(xmin,xmax, 500)

    pdfs = []
    for kde in kdes:
        pdfs.append(kde(x_grid))

    plt.figure(figsize=(8, 5))
    linestyles = ['-','--','-.',':']
    for pdf, linestyle, contract in zip(pdfs, linestyles, contracts):
        plt.plot(x_grid, pdf, label = f"{contract} contract", linewidth=2, linestyle=linestyle)

    plt.xlabel('Utility')
    plt.ylabel('Probability density')
    plt.title('Utility distributions')
    plt.legend()
    plt.tight_layout()
    plt.show()

def kde_plot(df):

    # ----------------------------
    # 1.  Get the two utility series
    # ----------------------------
    u_vmx  = df['Utility_VMX'].dropna().to_numpy()
    u_chen = df['Utility_Chen'].dropna().to_numpy()
    u_ni = df['Utility_NI'].dropna().to_numpy()

    # ----------------------------
    # 2.  Build KDEs
    # ----------------------------
    kde_vmx  = gaussian_kde(u_vmx)
    kde_chen = gaussian_kde(u_chen)
    kde_ni = gaussian_kde(u_ni)

    # ----------------------------
    # 3.  Evaluate across a common grid
    # ----------------------------
    xmin = -3
    xmax = max(u_vmx.max(),  u_chen.max())
    x_grid = np.linspace(xmin, xmax, 500)

    pdf_vmx  = kde_vmx(x_grid)
    pdf_chen = kde_chen(x_grid)
    pdf_ni = kde_ni(x_grid)

    # ----------------------------
    # 4.  Plot
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, pdf_vmx,  label='VMX contract',  linewidth=2)
    plt.plot(x_grid, pdf_chen, label='Chen contract', linewidth=2, linestyle='--')
    plt.plot(x_grid, pdf_ni, label='Chen contract', linewidth=2, linestyle='-o')

    plt.xlabel('Utility')
    plt.ylabel('Probability density')
    plt.title('Utility distributions: VMX vs. Chen')
    plt.legend()
    plt.tight_layout()
    plt.show()

def deep_dive():
    w_0 = 0.1
    alpha = 1.5
    c_k = 0.02
    mz = False
    if mz:
        ldf = load_loss_data()
        zone_sizes = ldf.groupby(['Zone','Year'])['Weight'].sum().groupby('Zone').mean().loc[ZONES]
        premium_kwargs = {'zone_sizes':zone_sizes}
        v_premium_kwargs = premium_kwargs.copy()
        v_premium_kwargs['cap_shares'] = True
        method = 'VMX-M'
    else:
        premium_kwargs = None
        v_premium_kwargs = None
        method = 'VMX'

    cdf = load_payouts('Chen', c_k, w_0, alpha)
    vdf = load_payouts(method,c_k, w_0, alpha)
    chdf = load_payouts('Chantarat', c_k, w_0, alpha)

    cdf = create_ii_df(cdf, c_k, premium_kwargs)
    vdf = create_ii_df(vdf, c_k, v_premium_kwargs)
    chdf = create_ii_df(chdf, c_k, premium_kwargs)

    cdf = add_utility_metrics(cdf, w_0, alpha)
    vdf = add_utility_metrics(vdf, w_0, alpha)
    chdf = add_utility_metrics(chdf, w_0, alpha)

    vdf['DeltaU'] = vdf['Utility'] - vdf['NI_Utility']
    cdf['DeltaU'] = cdf['Utility'] - cdf['NI_Utility']

    chdf['Utility_Chantarat'] = chdf['Utility']
    chdf['Payout_Chantarat'] = chdf['Payout']

    df = cdf.merge(vdf, on='Idx',suffixes=('_Chen','_VMX'))
    df = df.merge(chdf[['Utility_Chantarat','Payout_Chantarat','Idx']],on='Idx')
    df['Loss'] = df['Loss_Chen']
    df['Utility_NI'] = df['NI_Utility_VMX']
    cols = ['Loss','Payout_VMX','Payout_Chen','Utility_VMX','Utility_Chen','Utility_NI','Utility_Chantarat','Payout_Chantarat']
    df = df.loc[:, cols]

    ldf = df.loc[df.Loss > 0,:]
    ldf['LossQ'] = pd.qcut(ldf.Loss, 4,labels=False)

    ndf = df.loc[df.Loss == 0,:]

def sz_vs_mz_deep_dive():
    w_0 = 0.1
    alpha = 1.5
    c_k = 0.02
    ldf = load_loss_data()
    zone_sizes = ldf.groupby(['Zone','Year'])['Weight'].sum().groupby('Zone').mean().loc[ZONES]
    premium_kwargs = {'zone_sizes':zone_sizes}
    v_premium_kwargs = premium_kwargs.copy()
    v_premium_kwargs['cap_shares'] = True

    sdf = load_payouts('VMX', c_k, w_0, alpha)
    mdf = load_payouts('VMX-M', c_k, w_0, alpha)
    cdf = load_payouts('Chen',c_k, w_0, alpha)
    chdf = load_payouts('Chantarat',c_k,w_0,alpha)

    sdf = create_ii_df(sdf, c_k, None)
    mdf = create_ii_df(mdf, c_k, v_premium_kwargs)
    cdf = create_ii_df(cdf, c_k, premium_kwargs)
    chdf = create_ii_df(chdf, c_k, premium_kwargs)

    sdf = add_utility_metrics(sdf, w_0, alpha)
    mdf = add_utility_metrics(mdf, w_0, alpha)
    cdf = add_utility_metrics(cdf, w_0, alpha)
    chdf = add_utility_metrics(chdf, w_0, alpha)

    sdf['DeltaU'] = sdf['Utility'] - sdf['NI_Utility']
    mdf['DeltaU'] = mdf['Utility'] - mdf['NI_Utility']

    df = sdf.merge(mdf[['Payout','Utility','DeltaU','Idx']], on='Idx', suffixes=('_VMX','_VMX-M'))
    odf = cdf.merge(chdf[['Payout','Utility','Idx']],on='Idx',suffixes=('_Chen','_Chantarat'))
    cols = ['Loss','Payout_VMX','Payout_VMX-M','Utility_VMX','Utility_VMX-M','NI_Utility']
    df = df.merge(odf,on='Idx')
    # df = df.loc[:,cols]
    df['MZOverSZ'] = df['Utility_VMX-M'] > df['Utility_VMX']
    df['MZOverNI'] = df['Utility_VMX-M'] > df['NI_Utility']
    df['SZOverNI'] = df['Utility_VMX'] > df['NI_Utility']

    df['WMZOverSZ'] = df['Weight']*df['MZOverSZ']
    df['WMZOverNI'] = df['Weight']*df['MZOverNI']

    df['WUtility_VMX'] = df['Weight']*df['Utility_VMX']
    df['WUtility_VMX-M'] = df['Weight']*df['Utility_VMX-M']
    df['WUtility_NI'] = df['Weight']*df['NI_Utility']

    # Here, we'll show that a majority of farmers are better off with the MZ insurance vs SZ
    # The numbers go slightly up when we do that with the loss observations
    df['WMZOverSZ'].sum()/df['Weight'].sum()
    df['WMZOverNI']/sum()/df['Weight'].sum()

    # Consider showing weighted loss shares, NE3 accounts for nearly 60% of weighted losses.

    ldf = df.loc[df.Loss > 0,:].copy()
    tst = df.groupby('Zone')[['WMZOverSZ','WMZOverNI','Weight']].sum()
    tst['MZOverSZ'] = tst['WMZOverSZ']/tst['Weight']
    tst['MZOverNI'] = tst['WMZOverNI']/tst['Weight']

    tst = ldf.groupby(['Zone','LossQ'])[['WMZOverSZ','WMZOverNI','Weight']].sum()
    tst['MZOverSZ'] = tst['WMZOverSZ']/tst['Weight']
    tst['MZOverNI'] = tst['WMZOverNI']/tst['Weight']

def sz_vs_mz_contract_plots():
    w_0 = 0.1
    alpha = 1.5
    c_k = 0.02
    ldf = load_loss_data()
    zone_sizes = ldf.groupby(['Zone','Year'])['Weight'].sum().groupby('Zone').mean().loc[ZONES]
    premium_kwargs = {'zone_sizes':zone_sizes}
    v_premium_kwargs = premium_kwargs.copy()
    v_premium_kwargs['cap_shares'] = True

    sdf = load_payouts('VMX', c_k, w_0, alpha)
    mdf = load_payouts('VMX-M', c_k, w_0, alpha)

    sdf = sdf.loc[sdf.Set == 'Train',:]
    mdf = mdf.loc[mdf.Set == 'Train',:]

    df = sdf.merge(mdf[['Payout','PredLoss','Utility','Idx']], on='Idx', suffixes=('_VMX','_VMX-M'))

    zones = list(dict.fromkeys(df['Zone'].dropna()))
    ncols = 3
    nrows = int(np.ceil(len(zones) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(4.5 * ncols, 3.6 * nrows),
                            sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, z in enumerate(zones):
        ax = axes[i]
        d = df[df['Zone'] == z]

        # VMX
        tmp = d[['PredLoss_VMX', 'Payout_VMX']].dropna().sort_values('PredLoss_VMX')
        if not tmp.empty:
            ax.scatter(tmp['PredLoss_VMX'], tmp['Payout_VMX'],
                    label='SZ', alpha=0.9)

        # VMX-M
        tmpm = d[['PredLoss_VMX-M', 'Payout_VMX-M']].dropna().sort_values('PredLoss_VMX-M')
        if not tmpm.empty:
            ax.scatter(tmpm['PredLoss_VMX-M'], tmpm['Payout_VMX-M'],
                    label='MZ', alpha=0.9)

        ax.set_title(f'Zone {z}')
        ax.grid(True)

    # Turn off any unused panels (if zones < nrows*ncols)
    for j in range(len(zones), nrows * ncols):
        axes[j].axis('off')

    # Axis labels (leftmost + bottom row to avoid clutter)
    for r in range(nrows):
        axes[r * ncols].set_ylabel('Payout')
    for c in range((nrows - 1) * ncols, nrows * ncols):
        if c < len(zones):
            axes[c].set_xlabel('Predicted loss')

    # One shared legend outside the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)

    fig.suptitle('Payout vs Predicted Loss by Zone (SZ vs MZ)', y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.show()

def sz_vs_mz_contracts_sns():
    w_0 = 0.1
    alpha = 1.5
    c_k = 0.02
    ldf = load_loss_data()
    zone_sizes = ldf.groupby(['Zone','Year'])['Weight'].sum().groupby('Zone').mean().loc[ZONES]
    premium_kwargs = {'zone_sizes':zone_sizes}
    v_premium_kwargs = premium_kwargs.copy()
    v_premium_kwargs['cap_shares'] = True

    sdf = load_payouts('VMX', c_k, w_0, alpha)
    mdf = load_payouts('VMX-M', c_k, w_0, alpha)

    sdf = sdf.loc[sdf.Set == 'Train',:]
    mdf = mdf.loc[mdf.Set == 'Train',:]

    df = sdf.merge(mdf[['Payout','PredLoss','Utility','Idx']], on='Idx', suffixes=('_VMX','_VMX-M'))

    pairs = [
         ("MZ", "PredLoss_VMX-M", "Payout_VMX-M"),
         ("SZ",   "PredLoss_VMX",   "Payout_VMX"),
        ]
    
    long_parts = []
    for method, xcol, ycol in pairs:
        tmp = df[["Zone", xcol, ycol]].rename(columns={xcol: "PredLoss", ycol: "Payout"}).copy()
        tmp["Contract"] = method
        long_parts.append(tmp)

    long_df = pd.concat(long_parts, ignore_index=True).dropna(subset=["PredLoss", "Payout"])
    long_df["Zone"] = long_df["Zone"].astype(str)
    
     # ----- 2) Plot with seaborn: one facet per Zone, two lines (VMX vs VMX-M) -----
    sns.set_theme(style="whitegrid")
    
    g = sns.relplot(
         data=long_df,
         x="PredLoss", y="Payout",
         hue="Contract", style="Contract",
         hue_order=["SZ", "MZ"], style_order=["SZ", "MZ"],
         col="Zone", col_wrap=3, kind='scatter',                    
         facet_kws=dict(sharex=True, sharey=True)
     )
    
    g.set_axis_labels("Predicted loss", "Payout")
    g.set_titles("Zone {col_name}")
    g._legend.set_title("Contract")
    sns.move_legend(g,'upper left',bbox_to_anchor=(0.8,0.45))
    plt.tight_layout()
    plt.show()

def capital_allocation_bar_chart():
    w_0 = 0.1
    alpha = 1.5
    c_k = 0.02
    ldf = load_loss_data()
    zone_sizes = ldf.groupby(['Zone','Year'])['Weight'].sum().groupby('Zone').mean().loc[ZONES]
    premium_kwargs = {'zone_sizes':zone_sizes}
    v_premium_kwargs = premium_kwargs.copy()
    v_premium_kwargs['cap_shares'] = True

    sdf = load_payouts('VMX', c_k, w_0, alpha)
    mdf = load_payouts('VMX-M', c_k, w_0, alpha)

    sdf = create_ii_df(sdf, c_k, None)
    mdf = create_ii_df(mdf, c_k, v_premium_kwargs)

    df = sdf.merge(mdf[['Payout','Utility','CapCost','Idx']], on='Idx', suffixes=('_VMX','_VMX-M'))

    loss_shares = (df.groupby('Zone')['WLoss'].sum()/df.WLoss.sum()).reset_index()
    mz_cap_shares = (df.groupby('Zone')['WCapCost_VMX-M'].sum()/df['WCapCost_VMX-M'].sum()).reset_index()
    sz_cap_shares = (df.groupby('Zone')['WCapCost_VMX'].sum()/df.WCapCost_VMX.sum()).reset_index()

    tst = loss_shares.merge(sz_cap_shares,on='Zone')
    tst = tst.merge(mz_cap_shares, on='Zone')

    plot_df = tst
    cols = [col for col in tst.columns if col != 'Zone']

    rename_map = {
        'WLoss': 'Loss Share',
        'WCapCost_VMX': 'Capital Share SZ',
        'WCapCost_VMX-M': 'Capital Share MZ',  # note the hyphenated column name
    }
    value_cols = list(rename_map.keys())


    # long/tidy format for seaborn
    long_df = plot_df.melt(id_vars='Zone',
                        value_vars=value_cols,
                        var_name='Metric',
                        value_name='Value')
    long_df['Metric'] = long_df['Metric'].map(rename_map)

    # consistent legend/order
    metric_order = [rename_map['WLoss'], rename_map['WCapCost_VMX'], rename_map['WCapCost_VMX-M']]

    # --- plot ---
    sns.set_theme(style='whitegrid')
    ax = sns.barplot(data=long_df, x='Zone', y='Value',
                    hue='Metric', hue_order=metric_order,
                    dodge=True, errorbar=None)

    ax.set_xlabel('Zone')
    ax.set_ylabel('Share')
    ax.set_title('Loss & Capital Shares by Zone')
    ax.legend(title='', ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()

def payout_by_loss_quartile(df, n_buckets):
    df = df.loc[df.Loss > 0,:].copy()
    df['LossQ'] = pd.qcut(df.Loss, n_buckets, labels=False)
    df = df.groupby('LossQ')[['Payout_VMX','Payout_Chen']].quantile(q=np.arange(0.5,0.95,0.05)).reset_index()
    df['quantile'] = df['level_1']

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5), sharey=True)

    # Map column names → nicer titles
    contracts = {"Payout_VMX": "VMX contract",
                "Payout_Chen": "Chen contract"}

    # Optional: consistent colour palette for quartiles
    colors = {q: c for q, c in zip(sorted(df["LossQ"].unique()),
                                plt.cm.tab10(range(n_buckets)))}

    # -------------------------------------------
    # 2.  Loop over the two contracts
    # -------------------------------------------
    for ax, (col, title) in zip(axes, contracts.items()):
        # Draw one line per LossQ
        for q, grp in df.groupby("LossQ"):
            grp_sorted = grp.sort_values("quantile")          # ensure left→right
            ax.plot(grp_sorted["quantile"],
                    grp_sorted[col],
                    label=f"Quartile {q+1}", 
                    marker="o",
                    color=colors[q])

        # Axes styling
        ax.set_title(title)
        ax.set_xlabel("Payout quantile")
        ax.grid(True, axis="y")

    # Shared y-label on the left plot only
    axes[0].set_ylabel("Payout")

    # Common legend (or drop `bbox_to_anchor` & remove `loc` for per-plot legends)
    axes[1].legend(title="Loss quartile",
                bbox_to_anchor=(1.05, 0.5), loc="center left")

    fig.tight_layout()
    plt.show()

def quantile_function_plot(df):
    # columns to include
    cols = ['Payout_VMX', 'Payout_VMX-M','Payout_Chen', 'Payout_Chantarat', 'Loss']
    labels = ['VMX','VMX-M','Chen','Chantarat','Loss']

    # cols = ['Payout_VMX', 'Payout_VMX-M', 'Loss']
    # labels = ['VMX','VMX-M','Loss']

    df = df[cols].quantile(np.arange(0,1,0.01))

    # probability grid (e.g. every 0.5 %)
    q_grid = np.linspace(0, 1, 201)          # 0.00, 0.005, …, 1.00

    fig, ax = plt.subplots(figsize=(8, 5))

    for col, label in zip(cols,labels):
        data = df[col].dropna().to_numpy()
        # empirical quantiles at each probability in q_grid
        q_values = np.quantile(data, q_grid, interpolation='nearest')
        ax.plot(q_grid, q_values, label=label)

    ax.set_xlabel('Quantile (probability)')
    ax.set_ylabel('Value')
    ax.set_title('Empirical Quantile Function\n(Payouts and Loss)')
    ax.grid(True, axis='both')
    ax.legend(title='Column')
    plt.tight_layout()
    plt.show()



##### Deep Dive #####
# Part 1: Performance breakdown by zone

w_0 = 0.1
method = 'VMX'
alpha = 1.5
c_k = 0.02

# df = create_table(c_k,w_0,alpha)
# fname = f"ck{c_k}_r{alpha}_w{w_0}".replace('.','')
# df.to_latex(f"experiments/evaluation/Thailand/Test/{fname}_results.tex",index=False, float_format='%.3f')
# for alpha in [1.1, 1.5, 2, 2.5, 3, 3.5]:
#     for c_k in [0,0.02,0.04,0.05,0.06]:
#         print(f"Alpha {alpha} C_k {c_k}")
#         create_table(c_k,w_0,alpha)
        # create_mz_table(c_k, w_0, alpha)



create_table(c_k,w_0,alpha)
# create_mz_table(c_k, w_0, alpha)