import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# TODO: create a figure with total payout per length and average losses on the training set per length

# Input files/dirs

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
EVAL_DIR = os.path.join(EXPERIMENTS_DIR,'evaluation')
PREDICTIONS_DIR = os.path.join(EXPERIMENTS_DIR,'prediction')
FIGURES_DIR = os.path.join(PROJECT_DIR,'output','figures','Evaluation')

##### Data Loading #####
def load_model_predictions(state, length, model_name):
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

def get_eval_name(state, length, market_loading):
    length = str(length)
    pred_dir = os.path.join(EVAL_DIR,state,'Test')
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    rdf = rdf.loc[(rdf['Market Loading'] == market_loading) & (rdf['Method'] == 'Our Method'),:]
    idx = rdf['Utility'].idxmax()
    best_model = rdf.loc[idx, 'Eval Name']
    return best_model

def get_best_model(state, length):
    length = str(length)
    pred_dir = os.path.join(EVAL_DIR,state,'Val')
    results_fname = os.path.join(pred_dir,f"results_{length}.csv")
    rdf = pd.read_csv(results_fname)
    idx = rdf['Utility'].idxmax()
    best_model = rdf.loc[idx, 'Eval Name']
    return '_'.join(best_model.split('_')[:3])

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

state = 'Iowa'
iadf = get_raw_payouts(state)
sns.pointplot(data=iadf,x='Length',y='Utility',hue='Method',dodge=True)
plot_name = f"{state}_Average_Utility.png"
plot_fname = os.path.join(FIGURES_DIR,plot_name+'.png')
plt.savefig(plot_fname)
plt.close()

sns.pointplot(data=iadf,x='Length',y='Utility',hue='Method',dodge=True)
plot_name = f"{state}_Average_Utility.png"
plot_fname = os.path.join(FIGURES_DIR,plot_name+'.png')
plt.savefig(plot_fname)


state = 'Iowa'
rdf = get_results(state)
metrics = ['Utility','UtilityImprovement','Cost Per Utility','Cost']
market_loadings = [1]
for market_loading in market_loadings:
    for metric in metrics:
        metric_vs_length_figure(rdf, state,market_loading, metric, 'Length')

state = 'Illinois'
length = 30
model_name = 'chen_Ridge_Bja_ub88_teO'
df = load_payouts(state, length, model_name)


params = {'c_k':0.13, 'market_loading':1,'constrained':'uc','lr':0.01}
cdf = load_chen_payouts('Illinois',30,params)
cdf = cdf.loc[cdf.Set == 'Test',:]
cdf['Wealth'] = 388.6 -87.2 -cdf['Loss'] + cdf['Payout']
df['Method'] = 'Our Method'
cdf['Method'] = 'Chen'

bdf = pd.concat([df,cdf],ignore_index=True)
sns.lmplot(data=bdf,x='Loss',y='Payout',hue='Method')

params = {'c_k':0, 'market_loading':1.2414,'constrained':'uc','lr':0.001}
cdf2 = load_chen_payouts('Illinois',30,params)
cdf2 = cdf2.loc[cdf2.Set == 'Test',:]

sns.lmplot(data=bdf,x='Loss',y='Payout',hue='Method',palette=['tab:orange','tab:blue'])

our_payouts = []
chen_payouts = []
for i in range(100):
    ccdf2 = cdf2.sample(n=500)
    ccdf = cdf.sample(n=500)
    our_payouts.append(ccdf.Payout.sum())
    chen_payouts.append(ccdf2.Payout.sum())

plt.hist(chen_payouts,alpha=0.6,bins=30,label='Chen')
plt.hist(our_payouts,bins=30,label='Our method',alpha=0.6)
plt.xlabel('Insurer Cost')
plt.ylabel('Frequency')
# plt.hist(odf['TotalPayout'],alpha=0.5,bins=30,label='our method')
# plt.hist(bdf['TotalPayout'],alpha=0.5,bins=30,label='baseline')
# plt.xlabel('Insurer Cost')
# plt.ylabel('Frequency')
plt.legend()