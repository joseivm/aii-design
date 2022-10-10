import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import time

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

##### Optimization Functions #####
def min_CVaR_program(pred_y,train_y,params):
    # params: epsilon, pi_max, pi_min, beta_z, eps_p, c_k, K_z
    eps_p = params['epsilon_p']
    Ks = params['K']
    epsilon = params['epsilon']
    c_k = params['c_k']
    budget = params['B']
    premium_income_sq = params['premium income']

    n_samples = train_y.shape[0]
    n_zones = train_y.shape[1]
    Ks = np.tile(Ks,(n_samples,1))
    p = np.ones(n_samples)/n_samples
    A = cp.Variable((n_samples,n_zones))
    B = cp.Variable((n_samples,n_zones))
    t = cp.Variable((n_samples,n_zones))
    t_p = cp.Variable(n_samples)
    alpha = cp.Variable((n_samples,n_zones))
    gamma = cp.Variable((n_samples,n_zones))
    gamma_p = cp.Variable(n_samples)
    m = cp.Variable()
    K_p = cp.Variable()

    constraints = []

    # objective, m >= CVaR(l_z - I_z(theta_z))
    constraints.append(t[0,:] + (1/epsilon)*(p @ gamma) <= m*np.ones(n_zones))

    # CVaR constraints for each zone's loss, gamma^k_z >= l_z - min(a_z \hat{l_z}+b_z, K_z) - t
    constraints.append(gamma >= train_y - cp.multiply(pred_y,A) -B -t)
    constraints.append(gamma >= train_y - Ks -t)
    constraints.append(gamma >= 0)

    # Portfolio CVaR constraint: CVaR(\sum_z I_z(\theta_z)) <= K^P + \sum_z E[I_z(\theta_Z)]
    constraints.append(t_p + (1/eps_p)*(p @ gamma_p) <= K_p + premium_income_sq)
    constraints.append(gamma_p >= cp.sum(alpha,axis=1)-t_p)
    constraints.append(gamma_p >= 0)
    constraints.append(alpha >= cp.multiply(pred_y,A)+B)
    constraints.append(alpha >= 0)

    # budget constraint
    constraints.append(budget >= (1/n_samples)*cp.sum(alpha) + c_k*K_p)

    constraints.append(B <= 0)

    for i in range(n_samples-1):
        constraints.append(A[i,:] == A[i+1,:])
        constraints.append(B[i,:] == B[i+1,:])
        constraints.append(t[i,:] == t[i+1,:])

    objective = cp.Minimize(m)
    problem = cp.Problem(objective,constraints)
    # problem.solve(solver=cp.SCIPY, scipy_options={"method":"highs"})
    problem.solve(solver=cp.GUROBI)
    # print('Req capital: {}'.format(K_p.value))
    return (A.value[0,:], B.value[0,:])

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

    df.loc[df.Cluster == 'Upper','PredictedRate'] = upper_pred
    df.loc[df.Cluster == 'Lower','PredictedRate'] = lower_pred
    return df

##### Helper Functions ##### 
def construct_hhdfs_for_optimization(hhdf):
    upper_dfs = []
    lower_dfs = []
    for season in hhdf.Season.unique():
        uhdf = hhdf.loc[(hhdf.Cluster == 'Upper') & (hhdf.Season == season),:]
        lhdf = hhdf.loc[(hhdf.Cluster == 'Lower') & (hhdf.Season == season),:]

        min_size = np.minimum(uhdf.shape[0],lhdf.shape[0])
        upper_dfs.append(uhdf.sample(n=min_size))
        lower_dfs.append(lhdf.sample(n=min_size))
    
    uhdf = pd.concat(upper_dfs)
    lhdf = pd.concat(lower_dfs)
    return uhdf, lhdf

##### Baseline Method Functions #####
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

##### Evaluation Functions ##### 
def run_eval(train_data,test_data,hhdf,params):
    livestock_value = 150

    # Train prediction model for each cluster
    upper_model, lower_model = train_prediction_models(train_data)
    train_data = add_model_predictions(train_data,upper_model,lower_model)

    # Merge train data predictions with household data
    merge_cols = ['NDVILocation','Season']
    train_hhdf = hhdf.merge(train_data[['NDVILocation','Season','PredictedRate']],left_on=merge_cols,right_on=merge_cols)

    # Transform household predicted livstock loss into predicted monetary loss
    train_hhdf['PredictedLoss'] = train_hhdf['HerdSize']*train_hhdf['PredictedRate']*livestock_value
    train_hhdf['ActualLoss'] = train_hhdf['Losses']*livestock_value

    # Create dataset that has samples of the form (pred_loss_u, pred_loss_l) (actual_loss_u,actual_loss_l)
    uhdf, lhdf = construct_hhdfs_for_optimization(train_hhdf)
    upper_preds = uhdf['PredictedLoss']
    upper_actual = uhdf['ActualLoss']
    lower_preds = lhdf['PredictedLoss']
    lower_actual = lhdf['ActualLoss']
    pred_y = np.vstack([upper_preds,lower_preds]).T
    train_y = np.vstack([upper_actual,lower_actual]).T

    # Use dataset to determine baseline contract parameters and budget
    strike_vals = determine_strike_values(train_y,pred_y)
    params['B'], params['premium income'] = determine_budget_params(pred_y,strike_vals,params['K'],params['c_k'])

    # Plug in dataset into opt model to find contract parameters
    a,b = np.around(min_CVaR_program(pred_y,train_y,params),2)
    print('a: {}, b: {}'.format(a,b))

    # Use prediction model to predict on test data
    test_data = add_model_predictions(test_data,upper_model,lower_model)

    # Merge test data predictions with household data
    merge_cols = ['NDVILocation','Season']
    test_hhdf = hhdf.merge(test_data[['NDVILocation','Season','PredictedRate']],left_on=merge_cols,right_on=merge_cols)

    # Transform household predicted livstock loss into predicted monetary loss
    test_hhdf['PredictedLoss'] = test_hhdf['HerdSize']*test_hhdf['PredictedRate']*livestock_value
    test_hhdf['ActualLoss'] = test_hhdf['Losses']*livestock_value

    test_uhdf, test_lhdf = construct_hhdfs_for_optimization(test_hhdf)
    test_hhdf = pd.concat([test_uhdf,test_lhdf])

    # Compute performance metrics
    bdf, odf = make_payout_dfs(test_hhdf,strike_vals,a,b,params)
    cdf = comparison_df(bdf,odf,params['epsilon'])
    return cdf, bdf, odf

def comparison_df(bdf,odf,cvar_eps=0.2):
    bdict = compute_performance_metrics(bdf,cvar_eps)
    odict = compute_performance_metrics(odf,cvar_eps)
    sdf = pd.DataFrame([bdict,odict])
    sdf['Model'] = ['Baseline','Opt']
    # sdf.drop(columns=['NetLoss Upper','NetLoss Lower','Payout CVaR'],inplace=True)
    sdf.drop(columns=['VaR Upper','VaR Lower'],inplace=True)
    return sdf

def compute_performance_metrics(df,cvar_eps=0.2):
    sdf = {}
    # sdf['CVaR Upper'] = CVaR(df[df.Cluster == 'Upper'],'Losses','NetLoss',cvar_eps)
    # sdf['CVaR Lower'] = CVaR(df[df.Cluster == 'Lower'],'Losses','NetLoss',cvar_eps)
    # sdf['Total CVaR'] = CVaR(df,'TotalLoss','TotalNetLoss',cvar_eps)
    # sdf['$|CVaR_2 - CVaR_1|$'] = np.abs(sdf['CVaR Upper']-sdf['CVaR Lower'])
    # sdf['Max CVaR'] = np.maximum(sdf['CVaR Upper'],sdf['CVaR Lower'])

    sdf['VaR Upper'] = df.loc[df.Cluster == 'Upper','NetLoss'].quantile(0.95)
    sdf['VaR Lower'] = df.loc[df.Cluster == 'Lower','NetLoss'].quantile(0.95)
    # sdf['Total CVaR'] = CVaR(df,'TotalLoss','TotalNetLoss',cvar_eps)
    sdf['$|VaR_2 - VaR_1|$'] = np.abs(sdf['VaR Upper']-sdf['VaR Lower'])
    sdf['Max VaR'] = np.maximum(sdf['VaR Upper'],sdf['VaR Lower'])

    sdf['NetLoss Upper'] = df.loc[df.Cluster == 'Upper','NetLoss'].mean()
    sdf['NetLoss Lower'] = df.loc[df.Cluster == 'Lower','NetLoss'].mean()
    # sdf['NetTotal'] = df['TotalNetLoss'].mean()
    sdf['$|L_2 - L_1|$'] = np.abs(sdf['NetLoss Upper']-sdf['NetLoss Lower'])

    col_names = ['$P(L > Q({}))$'.format(i) for i in [.60]]
    loss_quantiles = np.quantile(df.loc[df.Cluster == 'Lower','Losses'],[0.6])
    for col_name, loss_quantile in zip(col_names,loss_quantiles):
        sdf[col_name] = ((df.loc[df.Cluster == 'Upper','NetLoss'] > loss_quantile) | (df.loc[df.Cluster == 'Lower','NetLoss'] > loss_quantile)).mean()

    # sdf['Required Capital'] = CVaR(df,'TotalPayout','TotalPayout',0.01) - df['TotalPayout'].mean()    
    # sdf['Payout CVaR'] = CVaR(df,'TotalPayout','TotalPayout',0.01)
    # sdf['Average Cost'] = df['TotalPayout'].mean() + 0.05*sdf['Required Capital']
    sdf['Average_cost'] = df['Payout'].mean()
    return(sdf)

def CVaR(df,loss_col,outcome_col,epsilon=0.2):
    q = np.quantile(df[loss_col],1-epsilon)
    cvar = df.loc[df[loss_col] >= q,outcome_col].mean()
    return cvar

def make_payout_dfs(hhdf,strike_vals,a,b,params):
    clusters = ['Upper','Lower']
    Ks = params['K']
    bdfs = []
    odfs = []
    for i, cluster in enumerate(clusters):
        thdf = hhdf.loc[hhdf.Cluster == cluster,:]
        bdf = pd.DataFrame()
        bdf['Losses'] = thdf['ActualLoss']
        bdf['PredictedLosses'] = thdf['PredictedLoss']
        bdf['Payout'] = np.maximum(bdf['PredictedLosses']-strike_vals[i],0)
        bdf['Payout'] = np.minimum(bdf['Payout'],Ks[i])
        bdf['NetLoss'] = bdf['Losses'] - bdf['Payout']
        
        odf = pd.DataFrame()
        odf['Losses'] = thdf['ActualLoss']
        odf['PredictedLosses'] = thdf['PredictedLoss']
        odf['Payout'] = np.maximum(a[i]*odf['PredictedLosses']+b[i],0)
        odf['Payout'] = np.minimum(odf['Payout'],Ks[i])
        odf['NetLoss'] = odf['Losses']-odf['Payout']

        bdf['Cluster'] = cluster
        odf['Cluster'] = cluster
        bdfs.append(bdf)
        odfs.append(odf)
    bdf = pd.concat(bdfs,ignore_index=True)
    odf = pd.concat(odfs,ignore_index=True)
    return bdf, odf

def make_payout_dfs_bad(hhdf,strike_vals,a,b,params):
    clusters = ['Upper','Lower']
    Ks = params['K']
    bdf = pd.DataFrame()
    odf = pd.DataFrame()
    for i, cluster in enumerate(clusters):
        pred_col = 'Predicted Loss {}'.format(cluster)
        payout_col = 'Payout {}'.format(cluster)
        loss_col = 'Losses {}'.format(cluster)
        net_loss_col = 'NetLoss {}'.format(cluster)

        tdf = hhdf.loc[hhdf.Cluster == cluster,:]
        bdf[pred_col] = tdf['PredictedLoss']
        bdf[payout_col] = np.maximum(bdf[pred_col]-strike_vals[i],0)
        bdf[payout_col] = np.minimum(bdf[payout_col],Ks[i])
        bdf[loss_col] = tdf['ActualLoss']
        bdf[net_loss_col] = bdf[loss_col] - bdf[payout_col]

        odf[pred_col] = tdf['PredictedLoss']
        odf[payout_col] = np.minimum(a[i]*odf[pred_col]+b[i],Ks[i])
        odf[payout_col] = np.maximum(0,odf[payout_col])
        odf[loss_col] = tdf['ActualLoss']
        odf[net_loss_col] = odf[loss_col]-odf[payout_col]

    odf['TotalPayout'] = odf['Payout Upper'] + odf['Payout Lower']
    odf['TotalLoss'] = odf['Losses Upper'] + odf['Losses Lower']
    odf['TotalNetLoss'] = odf['NetLoss Upper'] + odf['NetLoss Lower']

    bdf['TotalPayout'] = bdf['Payout Upper'] + bdf['Payout Lower']
    bdf['TotalLoss'] = bdf['Losses Upper'] + bdf['Losses Lower']
    bdf['TotalNetLoss'] = bdf['NetLoss Upper'] + bdf['NetLoss Lower']
    return bdf, odf

def determine_budget_params(pred_y,strike_vals,Ks,c_k=0.15):
    bdf = pd.DataFrame()
    n_zones = pred_y.shape[1]
    for zone in np.arange(n_zones):
        pred_col = 'PredictedLosses{}'.format(zone)
        payout_col = 'Payout{}'.format(zone)
        bdf[pred_col] = pred_y[:,zone]
        bdf[payout_col] = np.maximum(bdf[pred_col]-strike_vals[zone],0)
        bdf[payout_col] = np.minimum(bdf[payout_col],Ks[zone])

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
    hh_data = pd.read_csv(kenya_hh_data_filename)
    cvar_eps = 0.1
    K = 40000
    params = {'epsilon':cvar_eps,'epsilon_p':0.01,'K':np.array([K,K]),'rho':np.array([0.5,0.5]),'c_k':0.15}
    years = [str(i) for i in range(2010,2014)]
    results = []
    for year in years:
        train_data = reg_data.loc[~reg_data.Season.str.contains(year),:]
        test_data = reg_data.loc[reg_data.Season.str.contains(year),:]
        cv_results,a,b = run_eval(train_data,test_data,hh_data,params)
        results.append(cv_results)

    rdf = pd.concat(results)
    rdf.groupby('Model').mean()

