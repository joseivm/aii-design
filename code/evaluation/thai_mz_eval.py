import pandas as pd
import os
import numpy as np
import time
import re
import copy
from pathlib import Path
import itertools

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
PAYOUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'evaluation','Thailand','Test','payouts')

ZONES = ['C2','C3','NE1','NE2','NE3','S1','S2','N2','N3']
# ZONES = ['NE1','NE2','NE3']
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
def create_table(prem, c_k, w_0, subsidy, alpha):
    metrics = ['Method','RIB','DeltaCE','Premium','Cost_II','Cost_PI']
    methods = ['VMX','Chantarat']
    rdfs = []
    for method in methods:
        rdf = get_results(method, c_k, w_0, subsidy, prem, alpha)
        rdfs.append(rdf)

    df = pd.DataFrame(rdfs)
    return df[metrics]

def get_results(method, c_k, w_0, subsidy=0, prem='train', alpha=3.5):
    pdf = load_payouts(method, c_k, w_0, alpha)
    train_df = pdf.loc[pdf.Set == 'Train',:]
    test_df = pdf.loc[pdf.Set == 'Test',:]
    for zone, year in itertools.product(train_df.Zone.unique(),train_df.TestYear.unique()):
        ztest = test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),:]
        ztrain = train_df.loc[(train_df.Zone == zone) & (train_df.TestYear == year),:]
        if prem == 'train':
            premium = calculate_premium(ztrain, c_k, subsidy)
        else:
            premium = calculate_premium(ztest, c_k, subsidy)
        test_df.loc[(test_df.Zone == zone) & (test_df.TestYear == year),'Premium'] = premium

    for zone in train_df.Zone.unique():
        zdf = test_df.loc[test_df.Zone == zone,:]
        rdf = performance_metrics(zdf, 0.25, alpha=alpha)
        # print(f"{zone}: DeltaCE: {rdf['DeltaCE']} Max: {rdf['MaxDeltaCE']} BetterOff: {rdf['BetterOff']}")

    results = performance_metrics(test_df, 0.25, alpha)
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

def performance_metrics(payout_df, w_0=0.5, alpha=1.5):
    edf = payout_df.copy()

    ce_ii = certainty_equivalent(edf.Loss, edf.Payout, edf.Premium.mean(), w_0=w_0, alpha=alpha)
    ce_ii_og = certainty_equivalent(edf.Loss, edf.PredLoss, edf.PredLoss.mean(), w_0=w_0, alpha=alpha)
    ce_ni = certainty_equivalent(edf.Loss, 0, 0, w_0=w_0, alpha=alpha)
    ce_pi = certainty_equivalent(edf.Loss, edf.Loss, edf.Loss.mean(), w_0=w_0, alpha=alpha)
    
    delta_ce = 100*(ce_ii - ce_ni)/ce_ni
    max_delta_ce = 100*(ce_pi - ce_ni)/ce_ni
    rib = np.nan if max_delta_ce == 0 else delta_ce/max_delta_ce

    utility_ii = CRRA_utility(edf.Loss, edf.Payout, edf.Premium.mean(), w_0=w_0, alpha=alpha)
    utility_ii_og = CRRA_utility(edf.Loss, edf.PredLoss, edf.PredLoss.mean(), w_0=w_0, alpha=alpha)
    utility_ni = CRRA_utility(edf.Loss, 0, 0, w_0=w_0,alpha=alpha)
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

def certainty_equivalent(y_true, y_pred, premium, weight=1 ,w_0=0.5, alpha=1.5, markup=0):
    edf = pd.DataFrame({'Loss':y_true, 'Payout':y_pred, 'Weight': weight})
    edf['Wealth'] = w_0 - (1+markup)*premium + 1 - edf['Loss'] + edf['Payout']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    edf['WUtility'] = edf['Utility']*edf['Weight']
    average_utility = edf['WUtility'].sum()/edf['Weight'].sum()
    certainty_equivalent = ((1-alpha)*(average_utility))**(1/(1-alpha))
    return certainty_equivalent

def CRRA_utility(y_true, y_pred, premium, weight=1, w_0=0.5, alpha=1.5, markup=0):
    edf = pd.DataFrame({'Loss':y_true, 'Payout':y_pred, 'Weight':weight})
    edf['Wealth'] = w_0 - (1+markup)*premium + 1 - edf['Loss'] + edf['Payout']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    edf['WUtility'] = edf['Utility']*edf['Weight']
    return edf['WUtility'].sum()/edf['Weight'].sum()

def pct_better_off(y_true, y_pred, premium, w_0=0.5, alpha=1.5, markup=0):
    edf = pd.DataFrame({'Loss':y_true, 'Payout':y_pred})
    edf['Wealth'] = w_0 - (1+markup)*premium + 1 - edf['Loss'] + edf['Payout']
    edf['Wealth_NI'] = w_0 + 1 - edf['Loss']
    edf['Utility'] = 1/(1-alpha)*edf['Wealth']**(1-alpha)
    edf['Utility_NI'] = 1/(1-alpha)*edf['Wealth_NI']**(1-alpha)
    edf['BetterOff'] = edf['Utility'] > edf['Utility_NI']
    return edf['BetterOff'].mean()

def calculate_premium(payout_df, c_k, subsidy): 
    payout_cvar = CVaR(payout_df,'Payout','Payout',0.01)
    average_payout = payout_df['Payout'].mean()
    required_capital = payout_cvar-average_payout
    premium = average_payout + c_k*required_capital
    return (1-subsidy)*premium

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

def debugging():
    w_0 = 0.1
    c_k = 0
    subsidy = 0
    alpha = 2.5
    df = load_payouts('VMX',c_k,0.25,3.5)
    df['Province'] = df.Tambon.apply(lambda x: x[1:3])
    df = add_utility_metrics(df, w_0, alpha)

    mdf = df.groupby('Zone')[['Weight','PI_Utility','NI_Utility']].agg({'PI_Utility':'mean', 
                                                                        'NI_Utility':'mean', 'Weight':'sum'})
    mdf['DeltaU'] = -100*(mdf['PI_Utility'] - mdf['NI_Utility'])/mdf['NI_Utility']
    print(mdf.sort_values('DeltaU'))

    mdf['WDelta'] = mdf['DeltaU']*mdf['Weight']
    print(mdf.WDelta.sum()/mdf.Weight.sum())


# TODO: fix PI premium calculation for c_k = 0.13
prem = 'train'
c_k = 0.07
w_0 = 0.1
subsidy = 0
# alpha = 3.5
for alpha in [1.1,1.5, 2, 2.5, 3, 3.5]:
    df = create_table(prem,c_k,w_0,subsidy,alpha)
    fname = f"ck{c_k}_w{w_0}_r{alpha}".replace('.','')
    df.to_csv(f"experiments/evaluation/Thailand/Test/{fname}_results.csv",index=False, float_format='%.3f')