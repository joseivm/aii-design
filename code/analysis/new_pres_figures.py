import pandas as pd
import os
import numpy as np
import sklearn.metrics as metrics
import time
import cvxpy as cp
import random, string
from pathlib import Path
import math

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
EVAL_DIR = os.path.join(EXPERIMENTS_DIR,'evaluation')
PREDICTIONS_DIR = os.path.join(EXPERIMENTS_DIR,'prediction')

##### Data Loading #####
def load_model_predictions(state, length, model_name):
    pred_dir = os.path.join(PREDICTIONS_DIR,state,f"predictions {length}")
    pred_file = os.path.join(pred_dir,f"{model_name}_preds.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

def load_chen_payouts(state, length, params):
    length = str(length)
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