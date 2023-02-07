import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import cvxpy as cp
import matplotlib.pyplot as plt
import os
from sdv.tabular import GaussianCopula

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
PROCESSED_DATA_DIR = PROJECT_DIR + '/data/processed'
kenya_reg_data_filename = PROCESSED_DATA_DIR + '/Kenya/kenya_reg_data.csv'
kenya_hh_data_filename = PROCESSED_DATA_DIR + '/Kenya/kenya_hh_data.csv'

# Output files/dirs
FIGURES_DIR = PROJECT_DIR + '/output/figures'
TABLES_DIR = PROJECT_DIR + '/output/tables'

# TODO: figure out how to better enforce the budget constraint, think about the fact that one region
# has a lot more losses than the other region. 

##### Contract Design Functions #####
# Opt
def min_CVaR_program2(pred_y,train_y,params,include_premium=False):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    eps_p = params['epsilon_p']
    sizes = params['P']
    epsilon = params['epsilon']
    c_k = params['c_k']
    budget = params['B']

    n_samples = train_y.shape[0]
    n_zones = train_y.shape[1]
    S = np.tile(sizes,(n_samples,1))
    p = np.ones(n_samples)/n_samples
    A = cp.Variable((n_samples,n_zones))
    B = cp.Variable((n_samples,n_zones))
    t = cp.Variable((n_samples,n_zones))
    t_k = cp.Variable(n_samples)
    pi = cp.Variable((n_samples,n_zones))
    alpha = cp.Variable((n_samples,n_zones))
    omega = cp.Variable((n_samples,n_zones))
    gamma = cp.Variable((n_samples,n_zones))
    gamma_p = cp.Variable(n_samples)
    m = cp.Variable()
    K = cp.Variable()

    constraints = []

    # objective, m >= CVaR(l_z - I_z(theta_z))
    constraints.append(t[0,:] + (1/epsilon)*(p @ gamma) <= m*np.ones(n_zones))

    # CVaR constraints for each zone's loss, gamma^k_z >= l_z - min(a_z \hat{l_z}+b_z, K_z) - t
    if include_premium:
        constraints.append(gamma >= train_y + pi - omega -t)
    else:
        constraints.append(gamma >= train_y - omega -t)

    constraints.append(gamma >= 0)
    constraints.append(B <= 0)

    # Portfolio CVaR constraint: CVaR(\sum_z I_z(\theta_z)) <= K^P + \sum_z E[I_z(\theta_Z)]
    constraints.append(t_k + (1/eps_p)*(p @ gamma_p) <= K + (1/n_samples)*cp.sum(cp.multiply(S,omega)))
    constraints.append(gamma_p >= cp.sum(cp.multiply(S,alpha),axis=1)-t_k)
    constraints.append(gamma_p >= 0)
    constraints.append(omega <= cp.multiply(pred_y,A)+B)
    constraints.append(omega <= 1)
    constraints.append(alpha >= cp.multiply(pred_y,A)+B)
    constraints.append(alpha >= 0)

    # budget constraint
    constraints.append(budget >= (1/n_samples)*cp.sum(alpha) + c_k*K)

    # premium definition 
    if include_premium:
        constraints.append(pi[0,:] == (1/n_samples)*cp.sum(alpha,axis=0) + (1/np.sum(sizes)*c_k*K))

    for i in range(n_samples-1):
        constraints.append(A[i,:] == A[i+1,:])
        constraints.append(B[i,:] == B[i+1,:])
        constraints.append(t[i,:] == t[i+1,:])
        if include_premium:
            constraints.append(pi[i,:] == pi[i+1,:])

    objective = cp.Minimize(m)
    problem = cp.Problem(objective,constraints)
    if 'Projects' in os.getcwd():
        problem.solve(solver=cp.GUROBI)
    else:
        problem.solve(solver=cp.SCIPY, scipy_options={"method":"highs"})
    return (A.value[0,:], B.value[0,:])

def min_CVaR_program(pred_y,train_y,params,include_premium=False):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    eps_p = params['epsilon_p']
    P = params['P']
    epsilon = params['epsilon']
    c_k = params['c_k']
    budget = params['B']

    n_samples = train_y.shape[0]
    n_zones = train_y.shape[1]
    P = np.tile(P,(n_samples,1))
    p = np.ones(n_samples)/n_samples
    A = cp.Variable((n_samples,n_zones))
    B = cp.Variable((n_samples,n_zones))
    t = cp.Variable((n_samples,n_zones))
    t_p = cp.Variable(n_samples)
    alpha = cp.Variable((n_samples,n_zones))
    alpha_bar = cp.Variable((n_samples,n_zones))
    omega = cp.Variable((n_samples,n_zones))
    gamma = cp.Variable((n_samples,n_zones))
    gamma_p = cp.Variable(n_samples)
    m = cp.Variable()
    K_p = cp.Variable()

    constraints = []

    # objective, m >= CVaR(l_z - I_z(theta_z))
    constraints.append(t[0,:] + (1/epsilon)*(p @ gamma) <= m*np.ones(n_zones))

    # CVaR constraints for each zone's loss, gamma^k_z >= l_z - min(a_z \hat{l_z}+b_z, K_z) - t
    if include_premium:
        constraints.append(gamma >= train_y + alpha_bar - cp.multiply(pred_y,A) -B -t)
        constraints.append(gamma >= train_y + alpha_bar - P -t)
    else:
        constraints.append(gamma >= train_y - cp.multiply(pred_y,A) -B -t)
        constraints.append(gamma >= train_y - P -t)
    
    constraints.append(gamma >= 0)

    # Portfolio CVaR constraint: CVaR(\sum_z I_z(\theta_z)) <= K^P + \sum_z E[I_z(\theta_Z)]
    constraints.append(t_p + (1/eps_p)*(p @ gamma_p) <= K_p + (1/n_samples)*cp.sum(omega))
    constraints.append(omega <= cp.multiply(pred_y,A)+B)
    constraints.append(omega <= P)
    constraints.append(gamma_p >= cp.sum(alpha,axis=1)-t_p)
    constraints.append(gamma_p >= 0)
    constraints.append(alpha >= cp.multiply(pred_y,A)+B)
    constraints.append(alpha >= 0)

    # budget constraint
    constraints.append(budget >= (1/n_samples)*cp.sum(alpha) + c_k*K_p)

    # alpha_bar definition constraint
    if include_premium:
        constraints.append(alpha_bar[0,:] == (1/n_samples)*cp.sum(alpha,axis=0))

    constraints.append(B <= 0)

    for i in range(n_samples-1):
        constraints.append(A[i,:] == A[i+1,:])
        constraints.append(B[i,:] == B[i+1,:])
        constraints.append(t[i,:] == t[i+1,:])
        if include_premium:
            constraints.append(alpha_bar[i,:] == alpha_bar[i+1,:])

    objective = cp.Minimize(m)
    problem = cp.Problem(objective,constraints)
    # problem.solve(solver=cp.SCIPY, scipy_options={"method":"highs"})
    problem.solve(solver=cp.GUROBI)
    return (A.value[0,:], B.value[0,:])

# Baseline
def determine_strike_values(train_y,pred_y):
    num_zones = pred_y.shape[1]
    best_strike_vals = []
    best_strike_percentiles = []
    pred_losses = pred_y
    for zone in range(num_zones):
        strike_percentiles = np.arange(0.1,0.35,0.05)
        strike_vals = np.quantile(train_y[:,zone],strike_percentiles)
        strike_performance = {}

        for strike_percentile, strike_val in zip(strike_percentiles,strike_vals):
            strike_percentile = np.around(strike_percentile,2)
            insured_loss = np.maximum(train_y[:,zone]-strike_val,0)
            payout = np.maximum(pred_losses[:,zone]-strike_val,0).reshape(-1,1)
            loss_share_model = LinearRegression().fit(payout,insured_loss)
            share_explained = loss_share_model.coef_[0]
            strike_performance[(strike_percentile,strike_val)] = share_explained

        best_strike_percentile, best_strike_val = max(strike_performance,key=strike_performance.get)
        best_strike_vals.append(best_strike_val)
        best_strike_percentiles.append(best_strike_percentile)
    return best_strike_vals

##### Prediction Functions #####
def train_prediction_models(df):
    udf = df.loc[df.Cluster == 'Upper',:]
    ldf = df.loc[df.Cluster == 'Lower',:]

    train_x = udf[['PosZNDVI','NegZNDVI','PreNDVI','Good Regime']]
    train_y = udf['MortalityRate']
    upper_model = LinearRegression().fit(train_x,train_y)

    train_x = ldf[['PosZNDVI','NegZNDVI','PreNDVI','Good Regime']]
    train_y = ldf['MortalityRate']
    lower_model = LinearRegression().fit(train_x,train_y)
    return upper_model, lower_model

def add_model_predictions(df,upper_model, lower_model):
    udf = df.loc[df.Cluster == 'Upper',:]
    ldf = df.loc[df.Cluster == 'Lower',:]

    upper_X = udf[['PosZNDVI','NegZNDVI','PreNDVI','Good Regime']]
    upper_pred = upper_model.predict(upper_X)

    lower_X = ldf[['PosZNDVI','NegZNDVI','PreNDVI','Good Regime']]
    lower_pred = lower_model.predict(lower_X)

    df.loc[df.Cluster == 'Upper','PredMortalityRate'] = upper_pred
    df.loc[df.Cluster == 'Lower','PredMortalityRate'] = lower_pred
    return df

##### Helper Functions ##### 
def construct_data_for_optimization(hhdf):
    dfs = []
    for season in hhdf.Season.unique():
        uhdf = hhdf.loc[(hhdf.Cluster == 'Upper') & (hhdf.Season == season),:]
        lhdf = hhdf.loc[(hhdf.Cluster == 'Lower') & (hhdf.Season == season),:]

        combined_df = pd.merge(uhdf,lhdf,how='cross',suffixes=('_upper','_lower'))
        dfs.append(combined_df)
    
    df = pd.concat(dfs)
    pred_y = df[['PredMortalityRate_upper','PredMortalityRate_lower']].to_numpy()
    train_y = df[['MortalityRate_upper','MortalityRate_lower']].to_numpy()
    return pred_y,train_y

##### Performance Metric Functions #####
def comparison_df(bdf,odf,cvar_eps=0.2):
    bdict = compute_performance_metrics(bdf,cvar_eps)
    odict = compute_performance_metrics(odf,cvar_eps)
    sdf = pd.DataFrame([bdict,odict])
    sdf['Model'] = ['Baseline','Opt']
    # sdf.drop(columns=['NetLoss Upper','NetLoss Lower','Payout CVaR'],inplace=True)
    sdf.drop(columns=['VaR Upper','VaR Lower','CVaR Upper','CVaR Lower','SemiVar Upper','SemiVar Lower'],inplace=True)
    return sdf

def compute_performance_metrics(df,eps=0.2):
    sdf = {}
    sdf['CVaR Upper'] = CVaR(df[df.Cluster == 'Upper'],'Losses','NetLoss',eps)
    sdf['CVaR Lower'] = CVaR(df[df.Cluster == 'Lower'],'Losses','NetLoss',eps)
    # sdf['Total CVaR'] = CVaR(df,'TotalLoss','TotalNetLoss',cvar_eps)
    # sdf['$|CVaR_2 - CVaR_1|$'] = np.abs(sdf['CVaR Upper']-sdf['CVaR Lower'])
    sdf['Max CVaR'] = np.maximum(sdf['CVaR Upper'],sdf['CVaR Lower'])

    sdf['VaR Upper'] = df.loc[df.Cluster == 'Upper','NetLoss'].quantile(1-eps)
    sdf['VaR Lower'] = df.loc[df.Cluster == 'Lower','NetLoss'].quantile(1-eps)
    # sdf['Total CVaR'] = CVaR(df,'TotalLoss','TotalNetLoss',cvar_eps)
    sdf['$|VaR_2 - VaR_1|$'] = np.abs(sdf['VaR Upper']-sdf['VaR Lower'])
    sdf['Max VaR'] = np.maximum(sdf['VaR Upper'],sdf['VaR Lower'])

    sdf['SemiVar Upper'] = semi_variance(df[df.Cluster == 'Upper'],'Losses','NetLoss',eps)
    sdf['SemiVar Lower'] = semi_variance(df[df.Cluster == 'Lower'],'Losses','NetLoss',eps)
    sdf['Max SemiVar'] = np.maximum(sdf['SemiVar Upper'],sdf['SemiVar Lower'])

    # sdf['NetLoss Upper'] = df.loc[df.Cluster == 'Upper','NetLoss'].mean()
    # sdf['NetLoss Lower'] = df.loc[df.Cluster == 'Lower','NetLoss'].mean()
    # sdf['NetTotal'] = df['TotalNetLoss'].mean()
    # sdf['$|L_2 - L_1|$'] = np.abs(sdf['NetLoss Upper']-sdf['NetLoss Lower'])

    col_names = ['$P(L > Q({}))$'.format(i) for i in [.60]]
    loss_quantiles = np.quantile(df.loc[df.Cluster == 'Lower','ActualLoss'],[0.6])
    for col_name, loss_quantile in zip(col_names,loss_quantiles):
        sdf[col_name] = ((df.loc[df.Cluster == 'Upper','NetLoss'] > loss_quantile) | (df.loc[df.Cluster == 'Lower','NetLoss'] > loss_quantile)).mean()

    # sdf['Required Capital'] = CVaR(df,'TotalPayout','TotalPayout',0.01) - df['TotalPayout'].mean()    
    # sdf['Payout CVaR'] = CVaR(df,'TotalPayout','TotalPayout',0.01)
    # sdf['Average Cost'] = df['TotalPayout'].mean() + 0.05*sdf['Required Capital']
    sdf['Total Cost'] = df['Payout'].sum()
    return(sdf)

def calculate_premiums(train_hhdf,strike_vals,a,b,params):
    c_k = params['c_k']
    bdf, odf = make_payout_dfs(train_hhdf,strike_vals,a,b,params)
    bdf = bdf.groupby(['Season','Cluster'])['Payout'].sum().reset_index()
    bdf = bdf.pivot(index='Season',columns='Cluster',values='Payout').reset_index()
    baseline_copula = GaussianCopula()
    baseline_copula.fit(bdf[['Upper','Lower']])
    bdf = baseline_copula.sample(num_rows=10000)
    bdf['Total Payout'] = bdf['Upper'] + bdf['Lower']

    odf = odf.groupby(['Season','Cluster'])['Payout'].sum().reset_index()
    odf = odf.pivot(index='Season',columns='Cluster',values='Payout').reset_index()
    opt_copula = GaussianCopula()
    opt_copula.fit(odf[['Upper','Lower']])
    odf = opt_copula.sample(num_rows=10000)
    odf['Total Payout'] = odf['Upper']+ odf['Lower']
    bdf_cvar = CVaR(bdf,'Total Payout','Total Payout',0.01)
    odf_cvar = CVaR(odf,'Total Payout','Total Payout',0.01)

    baseline_average_total_payout = bdf['Total Payout'].mean()
    opt_average_total_payout = odf['Total Payout'].mean()

    baseline_req_capital = bdf_cvar-baseline_average_total_payout
    opt_req_capital = odf_cvar - opt_average_total_payout

    opt_average_payouts = odf[['Upper','Lower']].mean()
    baseline_average_payouts = bdf[['Upper','Lower']].mean()
    
    opt_premiums = opt_average_payouts + 0.5*c_k*opt_req_capital/10000
    baseline_premiums = baseline_average_payouts + 0.5*c_k*baseline_req_capital/10000
    return opt_premiums, baseline_premiums

def calculate_required_capital(train_hhdf,strike_vals,a,b,params):
    bdf, odf = make_payout_dfs(train_hhdf,strike_vals,a,b,params)
    baseline_average_payout = bdf.groupby('Season')['Payout'].sum().reset_index()['Payout'].mean()
    opt_average_payout = odf.groupby('Season')['Payout'].sum().reset_index()['Payout'].mean()
    bdf = bdf.groupby(['Season','Cluster'])['Payout'].sum().reset_index()
    bdf = bdf.pivot(index='Season',columns='Cluster',values='Payout').reset_index()
    baseline_copula = GaussianCopula()
    baseline_copula.fit(bdf[['Upper','Lower']])
    bdf = baseline_copula.sample(num_rows=10000)
    bdf['Total Payout'] = bdf['Upper'] + bdf['Lower']

    odf = odf.groupby(['Season','Cluster'])['Payout'].sum().reset_index()
    odf = odf.pivot(index='Season',columns='Cluster',values='Payout').reset_index()
    opt_copula = GaussianCopula()
    opt_copula.fit(odf[['Upper','Lower']])
    odf = opt_copula.sample(num_rows=10000)
    odf['Total Payout'] = odf['Upper']+ odf['Lower']
    bdf_cvar = CVaR(bdf,'Total Payout','Total Payout',0.01)
    odf_cvar = CVaR(odf,'Total Payout','Total Payout',0.01)

    baseline_req_capital = bdf_cvar-baseline_average_payout
    opt_req_capital = odf_cvar - opt_average_payout
    return baseline_req_capital, opt_req_capital

def make_payout_dfs(hhdf,strike_vals,a,b,params):
    clusters = ['Upper','Lower']
    Ps = params['P']
    bdfs = []
    odfs = []
    for i, cluster in enumerate(clusters):
        thdf = hhdf.loc[hhdf.Cluster == cluster,:]
        bdf = pd.DataFrame()
        bdf['ActualLoss'] = thdf['ActualLoss']
        bdf['PredictedLoss'] = thdf['PredictedLoss']
        bdf['Payout'] = np.maximum(bdf['PredictedLoss']-strike_vals[i],0)
        bdf['Payout'] = np.minimum(bdf['Payout'],Ps[i])
        bdf['NetLoss'] = bdf['ActualLoss'] - bdf['Payout']
        bdf['Season'] = thdf['Season']
        
        odf = pd.DataFrame()
        odf['ActualLoss'] = thdf['ActualLoss']
        odf['PredictedLoss'] = thdf['PredictedLoss']
        odf['Payout'] = np.maximum(a[i]*odf['PredictedLoss']+b[i],0)
        odf['Payout'] = np.minimum(odf['Payout'],Ps[i])
        odf['NetLoss'] = odf['ActualLoss']-odf['Payout']
        odf['Season'] = thdf['Season']

        bdf['Cluster'] = cluster
        odf['Cluster'] = cluster
        bdfs.append(bdf)
        odfs.append(odf)
    bdf = pd.concat(bdfs,ignore_index=True)
    odf = pd.concat(odfs,ignore_index=True)
    return bdf, odf

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

def semi_variance(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    semi_variance = (df.loc[df[loss_col] >= q,outcome_col]-q)**2
    return semi_variance.sum()/(len(semi_variance)-1)

##### Evaluation Functions ##### 
def run_eval(train_data,test_data,hhdf,params,include_premium=False,surplus_share=0.65):
    livestock_value = 150

    # Train prediction model for each cluster
    upper_model, lower_model = train_prediction_models(train_data)
    train_data = add_model_predictions(train_data,upper_model,lower_model)

    # Merge train data predictions with household data
    merge_cols = ['NDVILocation','Season']
    train_hhdf = hhdf.merge(train_data[['NDVILocation','Season','PredMortalityRate']],left_on=merge_cols,right_on=merge_cols)

    # Transform household predicted livstock loss into predicted monetary loss
    # train_hhdf['PredictedLoss'] = train_hhdf['HerdSize']*train_hhdf['PredMortalityRate']*livestock_value
    # train_hhdf['ActualLoss'] = train_hhdf['LivestockLosses']*livestock_value

    # Create dataset that has samples of the form (pred_loss_u, pred_loss_l) (actual_loss_u,actual_loss_l)
    pred_y, train_y = construct_data_for_optimization(train_hhdf)

    # Use dataset to determine baseline contract parameters and budget
    strike_vals = determine_strike_values(train_y,pred_y)
    params['B'], params['premium income'] = determine_budget_params(pred_y,strike_vals,params['P'],params['c_k'])

    # Plug in dataset into opt model to find contract parameters
    a,b = np.around(min_CVaR_program(pred_y,train_y,params,include_premium),2)

    # Use prediction model to predict on test data
    test_data = add_model_predictions(test_data,upper_model,lower_model)

    # Merge test data predictions with household data
    merge_cols = ['NDVILocation','Season']
    test_hhdf = hhdf.merge(test_data[['NDVILocation','Season','PredMortalityRate']],left_on=merge_cols,right_on=merge_cols)

    # Transform household predicted livstock loss into predicted monetary loss
    test_hhdf['PredictedLoss'] = test_hhdf['HerdSize']*test_hhdf['PredMortalityRate']*livestock_value
    test_hhdf['ActualLoss'] = test_hhdf['LivestockLosses']*livestock_value

    # test_uhdf, test_lhdf = construct_data_for_optimization(test_hhdf)
    # test_hhdf = pd.concat([test_uhdf,test_lhdf])
    # baseline_premium, opt_premium = calculate_premiums_copula()

    # Reallocate surplus and compute performance metrics 
    bdf, odf = make_payout_dfs(test_hhdf,strike_vals,a,b,params)
    odf = reallocate_opt_surplus(bdf,odf,params['epsilon'],surplus_share)
    cdf = comparison_df(bdf,odf,params['epsilon'])

    n_farmers = test_hhdf.shape[0]
    baseline_req_capital, opt_req_capital = calculate_required_capital(train_hhdf,strike_vals,a,b,params)
    cdf.loc[cdf.Model == 'Baseline','Required Capital'] = baseline_req_capital
    cdf.loc[cdf.Model == 'Opt','Required Capital'] = opt_req_capital
    cdf.loc[cdf.Model == 'Baseline','Average Cost'] = (cdf.loc[cdf.Model == 'Baseline','Total Cost'] + cdf.loc[cdf.Model == 'Baseline','Required Capital'])/n_farmers
    cdf.loc[cdf.Model == 'Opt','Average Cost'] = (cdf.loc[cdf.Model == 'Opt','Total Cost'] + cdf.loc[cdf.Model == 'Opt','Required Capital'])/n_farmers
    return cdf[['Model','Max CVaR','Max VaR','Max SemiVar','$|VaR_2 - VaR_1|$','Required Capital','Average Cost']], bdf, odf

def reallocate_opt_surplus(bdf,odf,eps,surplus_share=0.65):
    cdf = comparison_df(bdf,odf,eps)
    cost_diff = (cdf.loc[cdf.Model == 'Baseline','Total Cost'].to_numpy()[0] - cdf.loc[cdf.Model == 'Opt','Total Cost'].to_numpy()[0])*surplus_share
    loss_0_lb = odf.loc[odf.Cluster == 'Upper','PredictedLoss'].quantile(0.65)
    loss_0_ub = odf.loc[odf.Cluster == 'Upper','PredictedLoss'].quantile(0.85)
    loss_1_lb = odf.loc[odf.Cluster == 'Lower','PredictedLoss'].quantile(0.65)
    loss_1_ub = odf.loc[odf.Cluster == 'Lower','PredictedLoss'].quantile(0.85)

    zone_0_extra_payouts = (odf.PredictedLoss >= loss_0_lb) & (odf.PredictedLoss <= loss_0_ub)
    zone_0_extra_payouts = (zone_0_extra_payouts) & (odf.Cluster == 'Upper')
    zone_1_extra_payouts = (odf.PredictedLoss >= loss_1_lb) & (odf.PredictedLoss <= loss_1_ub)
    zone_1_extra_payouts = (zone_1_extra_payouts) & (odf.Cluster == 'Lower')

    per_farmer_extra_payout = cost_diff/(zone_0_extra_payouts.sum()+zone_1_extra_payouts.sum())
    odf.loc[zone_0_extra_payouts, 'Payout'] += per_farmer_extra_payout
    odf.loc[zone_1_extra_payouts, 'Payout'] += per_farmer_extra_payout
    odf['NetLoss'] = odf['ActualLoss'] - odf['Payout']
    return odf

def determine_budget_params(pred_y,strike_vals,Ps,c_k=0.15):
    bdf = pd.DataFrame()
    n_zones = pred_y.shape[1]
    for zone in np.arange(n_zones):
        pred_col = 'PredMortalityRate{}'.format(zone)
        payout_col = 'Payout{}'.format(zone)
        bdf[pred_col] = pred_y[:,zone]
        bdf[payout_col] = np.maximum(bdf[pred_col]-strike_vals[zone],0)
        bdf[payout_col] = np.minimum(bdf[payout_col],1)
        bdf[payout_col] = bdf[payout_col]*Ps[zone]

    bdf['Total Payouts'] = bdf['Payout0']+bdf['Payout1']
    loss_quantile = np.quantile(bdf['Total Payouts'],0.99)
    average_payout = bdf['Total Payouts'].mean()
    required_capital = loss_quantile-average_payout
    # print('Required Capital: {}'.format(required_capital))
    capital_costs = c_k*required_capital
    average_total_costs = average_payout + capital_costs
    return average_total_costs, average_payout

def cross_val():
    reg_data = pd.read_csv(kenya_reg_data_filename)
    hhdf = pd.read_csv(kenya_hh_data_filename)
    cvar_eps = 0.2
    years = [str(i) for i in range(2010,2014)]
    results = []
    for year in years:
        train_data = reg_data.loc[~reg_data.Season.str.contains(year),:]
        test_data = reg_data.loc[reg_data.Season.str.contains(year),:]
        average_sizes = train_data.groupby('Cluster')['HerdSize'].sum().to_numpy()[::-1]
        params = {'epsilon':cvar_eps,'epsilon_p':0.01,'P':average_sizes,'rho':np.array([0.5,0.5]),'c_k':0.15}
        cv_results,a,b = run_eval(train_data,test_data,hhdf,params,0.65)
        results.append(cv_results)

    rdf = pd.concat(results)
    res = rdf.groupby('Model').mean().reset_index()
    print(res)
    res_filename = TABLES_DIR + '/Kenya/kenya_eval2.tex'
    res.to_latex(res_filename,float_format='%.0f',escape=False,index=False)