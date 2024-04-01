import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce


from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# TODO: check that for each state and length we are using the same data as chen. For each state
# and legnth, make sure chen payouts are the same as in the single state case. Compare the premiums and
# see why it's slightly smaller in the multi state case for IL. Check that the single zone data matches the chen data

# Figures: for each state, compare our single zone, our multi zone, and chen. Compare 
# utility, insurer costs, and required capital. Distribution of total costs?

# For each state compare premiums and utility
# Distribution of total payouts?

# Input files/dirs

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
EVAL_DIR = os.path.join(EXPERIMENTS_DIR,'evaluation')
PREDICTIONS_DIR = os.path.join(EXPERIMENTS_DIR,'prediction')
TRANSFORMS_DIR = os.path.join(PROJECT_DIR,'data','time-series-transforms')
FIGURES_DIR = os.path.join(PROJECT_DIR,'output','figures','Midwest Evaluation')

# Figures: for each state, compare our single zone, our multi zone, and chen. 
# Distribution of total payouts?
states = ['Illinois','Indiana','Iowa','Missouri']
##### Data Loading #####
def load_model_predictions(state, length):
    model_name = get_best_model(state, length)
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

def load_all_payouts(states, length):
    pass

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

def get_best_model(state, length):
    length = str(length)
    pred_dir = os.path.join(EVAL_DIR,state,'Val')
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    idx = rdf['Utility'].idxmax()
    best_model = rdf.loc[idx, 'Eval Name']
    return '_'.join(best_model.split('_')[:3])

def load_all_single_zone_payouts(states, length):
    length = str(length)
    dfs = []
    for state in states:
        model_preds = get_payouts(state, length)
        pred_years = load_years(state, length,val=False)
        model_preds['State'] = state
        model_preds['CountyYear'] = pred_years
        model_preds['Year'] = model_preds['CountyYear'].apply(lambda x: int(x.split('-')[1]))
        dfs.append(model_preds)

    return pd.concat(dfs,ignore_index=True)

def get_payouts(state, length):
    model_name = get_eval_name(state, length, market_loading=1)
    df = load_payouts(state, length, model_name)
    return df

def get_results():
    lengths = [i*10 for i in range(2,9)]
    rdfs = []
    for length in lengths:
        fname = os.path.join(EVAL_DIR,'Midwest','Test',f"results_{length}.csv")
        rdf = pd.read_csv(fname)
        rdf['Length'] = length
        rdfs.append(rdf)

    rdf = pd.concat(rdfs)

    srdf = get_single_zone_results(states)
    return pd.concat([rdf,srdf],ignore_index=True)

def get_eval_name(state, length, market_loading):
    length = str(length)
    pred_dir = os.path.join(EVAL_DIR,state,'Test')
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    rdf = rdf.loc[(rdf['Market Loading'] == market_loading) & (rdf['Method'] == 'Our Method'),:]
    idx = rdf['Utility'].idxmax()
    best_model = rdf.loc[idx, 'Eval Name']
    return best_model
    
def get_single_zone_results(states):
    params = {'epsilon_p':0.01,'c_k':0.13,'subsidy':0,'w_0':388.6,
            'risk_coef':0.008,'S':[87,74,94,46], 'market_loading':1}
    lengths = [i*10 for i in range(2,9)]
    results = []
    for length in lengths:
        df = load_all_single_zone_payouts(states, length)
        train_df = df.loc[df.Set =='Train',:]
        test_df = df.loc[df.Set == 'Test',:]
        premiums, req_capital = calculate_premiums(train_df,params['c_k'],params['S'])
        eval_df = create_eval_df(test_df,premiums,params)
        metrics = calculate_performance_metrics(eval_df)
        metrics['Required Capital'] = req_capital
        metrics['Method'] = 'Our Method, SZ'
        metrics['Length'] = length
        results.append(metrics)

    return pd.DataFrame(results)

##### Eval functions #####
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

##### Midwest figures #####
# Compare our method, Chen's method, and our method in the single zone case. 
# Need to calculate premium for our method in the single zone case
# Utility, overall costs, required capital, profits?
# Need to read in midwest results, and i
def midwest_figures():
    rdf = get_results()
    metrics = ['Overall Utility','Insurer Cost','Required Capital']
    for metric in metrics:
        midwest_figure(rdf,metric,'Length')

def midwest_figure(rdf, metric, length):
    plot_name = f"Midwest_{metric}_{length}"
    sns.relplot(data=rdf,x=length,y=metric,hue='Method')
    plot_fname = os.path.join(FIGURES_DIR,plot_name+'.png')
    plt.savefig(plot_fname)


##### State figures #####
# Utility, premium, 
def state_figures():
    rdf = get_results()
    states = ['Illinois','Indiana','Iowa','Missouri']
    metrics = ['Premium','Utility']    
    for state in states: 
        for metric in metrics:
            state_figure(rdf,state,metric,'Length')

def state_figure(rdf,state, metric, length):
    plot_name = f"{state}_{metric}_{length}"
    y_col = f"{state}_{metric}"
    sns.relplot(data=rdf,x=length,y=y_col,hue='Method')
    plot_fname = os.path.join(FIGURES_DIR,plot_name+'.png')
    plt.savefig(plot_fname)
    

def farmer_wealth_histogram():
    plt.close()

    plt.hist(cdf['Wealth'],alpha=0.6,bins=30,label='Chen')
    plt.hist(df['Wealth'],bins=30,label='Our method',alpha=0.6)
    plt.xlabel('Farmer Wealth')
    plt.ylabel('Frequency')
    # plt.hist(odf['TotalPayout'],alpha=0.5,bins=30,label='our method')
    # plt.hist(bdf['TotalPayout'],alpha=0.5,bins=30,label='baseline')
    # plt.xlabel('Insurer Cost')
    # plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(fig_filename)

def get_chen_costs(state):
    market_loadings = [1]
    costs = []
    for i in range(2,9):
        length = str(i*10)
        for market_loading in market_loadings:
            df = load_chen_payouts(state, length, market_loading)
            market_loading = np.round(market_loading,3)
            length_cost = {'Length':int(length), 'Method': 'Chen uc', 'Market Loading':market_loading}
            length_cost['Cost'] = df.loc[df.Set == 'Test','Payout'].sum()
            costs.append(length_cost)

    return pd.DataFrame(costs)

def get_model_costs(state, rdf):
    costs = []
    rdf = rdf.loc[rdf.Method == 'Our Method',:]
    for idx, row in rdf.iterrows():
        df = load_payouts(state, row['Length'],row['Eval Name'])
        length_cost = {'Length':row['Length'], 'Method': 'Our Method','Market Loading':row['Market Loading']}
        length_cost['Insurer Cost'] = df.loc[df.Set == 'Test','Payout'].sum()
        costs.append(length_cost)
    return pd.DataFrame(costs)

def plot_training_set_loss(state):
    lengths = [i*10 for i in range(2,9)]
    data = []
    for length in lengths:
        pdf = load_chen_payouts(state,length,1)
        average_loss = pdf.loc[pdf.Set == 'Train','Loss'].mean()
        length_cost = {'Length':length,'Average Loss':average_loss,'State':state}
        data.append(length_cost)

    pdf = pd.DataFrame(data)
    sns.relplot(data=pdf,x='Length',y='Average Loss')
    plot_name = f"{state}_Train_Loss_length"
    plot_fname = os.path.join(FIGURES_DIR,plot_name+'.png')
    plt.savefig(plot_fname)
    # return pdf

def get_payouts(state, length, market_loading):
    model_name = get_eval_name(state, length, market_loading)
    df = load_payouts(state, length, model_name)
    return df

def get_results(state):
    lengths = [i*10 for i in range(2,9)]
    rdfs = []
    for length in lengths:
        fname = os.path.join(EVAL_DIR,state,'Test',f"results_{length}.csv")
        rdf = pd.read_csv(fname)
        rdf['Length'] = length
        rdfs.append(rdf)

    rdf = pd.concat(rdfs)
    rdf.loc[rdf.Method == 'No Insurance','Market Loading'] = 1.241
    ni_df = rdf.loc[rdf.Method == 'No Insurance',:].copy()
    ni_df['Market Loading'] = 1
    rdf = pd.concat([rdf,ni_df],ignore_index=True)

    ni_df = rdf.loc[rdf.Method == 'No Insurance',['Length','Utility','Market Loading']]
    rdf = rdf.merge(ni_df,on=['Length','Market Loading'],suffixes=('',' NI'))
    rdf['UtilityImprovement'] = rdf['Utility']-rdf['Utility NI']

    chen_costs = get_chen_costs(state)
    our_costs = get_model_costs(state,rdf)
    all_costs = pd.concat([chen_costs, our_costs])
    rdf = rdf.merge(all_costs, on=['Length','Market Loading','Method'],how='left')
    rdf['Cost Per Utility'] = rdf['Cost']/rdf['UtilityImprovement']
    return rdf

def get_prediction_results(metric):
    dfs = []
    states = ['Illinois','Indiana','Iowa','Missouri']
    for state in states:
        lengths = [i*10 for i in range(2,9)]
        rdfs = []
        for length in lengths:
            fname = os.path.join(EVAL_DIR,state,'Test',f"results_{length}.csv")
            rdf = pd.read_csv(fname)
            rdf['Length'] = length
            rdfs.append(rdf)

        rdf = pd.concat(rdfs)
        sdf = rdf.groupby('Length')[metric].max().reset_index(name=metric)
        sdf['State'] = state
        dfs.append(sdf)

    bdf = pd.concat(dfs,ignore_index=True)
    return bdf

def metric_vs_length_figure(rdf, state, market_loading, metric, length):
    premium = 'Our Premium' if market_loading == 1 else 'Chen Premium'
    plot_name = f"{state}_{metric}_{length}_ml{market_loading}".replace('.','')
    sns.relplot(data=rdf.loc[rdf['Market Loading'] == market_loading,:],x=length,y=metric,hue='Method')
    plot_fname = os.path.join(FIGURES_DIR,plot_name+'.png')
    plt.savefig(plot_fname)

def get_chen_premium(state, length, market_loading):
    fname = os.path.join(EVAL_DIR,state,'Test',f"results_{length}.csv")
    rdf = pd.read_csv(fname)
    premium = rdf.loc[(rdf['Market Loading'] == market_loading) & (rdf.Method == 'Chen uc'),'Premium'].item()
    return premium

def get_raw_payouts(state):
    market_loading = 1
    alpha = 0.008
    lengths = [i*10 for i in range(2,9)]
    dfs = []
    for length in lengths:
        odf = get_payouts(state, length, market_loading)
        odf = odf.loc[odf.Set == 'Test',:]
        cdf = load_chen_payouts(state, length, market_loading)
        nidf = pd.DataFrame()
        cdf = cdf.loc[cdf.Set == 'Test',:]

        nidf['Loss'] = cdf['Loss'].to_numpy()
        nidf['Wealth'] = 388.6 - nidf['Loss']
        nidf['Utility'] = -(1/alpha)*np.exp(-alpha*nidf['Wealth'])
        # nidf = pd.DataFrame({'Utility':nidf['Utility'].mean(),})

        cdf['Premium'] = get_chen_premium(state, length, market_loading)
        cdf['Wealth'] = 388.6 - cdf['Premium'] - cdf['Loss'] + cdf['Payout']
        cdf['Utility'] = -(1/alpha)*np.exp(-alpha*cdf['Wealth'])

        odf['Method'] = 'Our Method'
        cdf['Method'] = 'Chen'
        nidf['Method'] = 'No Insurance'
        odf['Length'] = length
        cdf['Length'] = length
        nidf['Length'] = length
        dfs += [odf,cdf,nidf]

    df = pd.concat(dfs,ignore_index=True)
    return df

def confidence_bar_plots(state):
    df = get_raw_payouts(state)
    sns.pointplot(data=df,x='Length',y='Utility',hue='Method',dodge=True)
    plot_name = f"{state}_Average_Utility_CI.png"
    plot_fname = os.path.join(FIGURES_DIR,plot_name)
    plt.savefig(plot_fname)
    plt.close()

    sns.pointplot(data=df,x='Length',y='Utility',hue='Method',dodge=True,estimator=np.median)
    plot_name = f"{state}_Median_Utility_CI.png"
    plot_fname = os.path.join(FIGURES_DIR,plot_name)
    plt.savefig(plot_fname)
    plt.close()

def over_under_prediction_errors_vs_length(state):
    lengths = [i*10 for i in range(2,9)]
    dfs = []
    for length in lengths:
        df = get_payouts(state, length, 1)
        df = df.loc[df.Set == 'Test',:]
        df['ErrorType'] = ''
        df.loc[df.PredLoss > df.Loss,'ErrorType'] = 'OverPrediction'
        df.loc[df.PredLoss < df.Loss, 'ErrorType'] = 'UnderPrediction'
        error_counts = (df.groupby('ErrorType').size()/df.shape[0]).reset_index(name='N')
        error_counts['Length'] = length
        dfs.append(error_counts)

    df = pd.concat(dfs)
    sns.barplot(data=df,x='Length',y='N',hue='ErrorType')
    plot_name = f"{state}_over_under_pred_erro_length"
    plot_fname = os.path.join(FIGURES_DIR,plot_name+'.png')
    plt.savefig(plot_fname)
