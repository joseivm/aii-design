import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

scrambled = False
suffix = '-scrambled' if scrambled else ''
# Input files/dirs

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
EVAL_DIR = os.path.join(EXPERIMENTS_DIR,f"evaluation{suffix}")
PRED_DIR = os.path.join(EXPERIMENTS_DIR,f"prediction{suffix}")
FIGURES_DIR = os.path.join(PROJECT_DIR,'output','figures','Proposal')

##### Data Loading #####
def get_results(state):
    rdfs = []
    result_dir = os.path.join(EVAL_DIR,state,'Test')
    fnames = [f for f in os.listdir(result_dir) if '_new.csv' in f]
    for fname in fnames:
        length = fname.split('_')[1]
        fpath = os.path.join(result_dir,fname)
        rdf = pd.read_csv(fpath)
        rdf['Length'] = length
        rdfs.append(rdf)

    rdf = pd.concat(rdfs)
    rdf['Length'] = rdf.Length.astype(int)
    return rdf

def get_pred_results(state):
    rdfs = []
    result_dir = os.path.join(PRED_DIR,state)
    fnames = [f for f in os.listdir(result_dir) if '.csv' in f]
    for fname in fnames:
        length = fname.split('_')[1][:-4]
        fpath = os.path.join(result_dir,fname)
        rdf = pd.read_csv(fpath)
        rdf['Length'] = length
        rdfs.append(rdf)

    rdf = pd.concat(rdfs)
    rdf['Length'] = rdf.Length.astype(int)
    return rdf

def utility_vs_length_quadratic(length=20):
    rdf = get_results('Illinois')
    # rdf = rdf.loc[rdf.Length > length,:]
    # rdf = rdf.loc[rdf.Length != 24,:]
    plt.figure()
    ax = sns.lmplot(
        data=rdf,
        x="Length",
        y="Utility",
        hue="Method",
        markers=["o", "s"],
        palette="Set2",
        height=6,
        aspect=1.2,
        scatter_kws={"alpha": 0.7, "s": 60},
        line_kws={"lw": 2.5},
        order=2                # 👈 Quadratic regression
    )
    sns.move_legend(ax,'upper right')
    plt.title("Utility vs. Data Length by Method (Quadratic Fit)", fontsize=14, pad=15)
    plt.xlabel("Training Data Length")
    plt.ylabel("Expected Utility")
    plt.tight_layout()
    plt.show()

def utility_vs_length_lowess(length=20):
    rdf = get_results('Illinois')
    rdf = rdf.loc[rdf.Length > length,:]
    ax = sns.lmplot(
        data=rdf,
        x="Length",
        y="Utility",
        hue="Method",
        lowess=True,           # 👈 Nonparametric locally weighted regression
        scatter_kws={"alpha": 0.7, "s": 60},
        line_kws={"lw": 2.5},
        palette="Set2",
        height=6,
        aspect=1.2
    )
    sns.move_legend(ax,'upper right',bbox_to_anchor=(0,0.25))
    plt.title("Utility vs. Data Length by Method (LOWESS Fit)", fontsize=14, pad=15)
    plt.xlabel("Training Data Length")
    plt.ylabel("Expected Utility")
    plt.tight_layout()
    plt.show()

def performance_vs_length():
    rdf = get_pred_results('Illinois')

    # Compute averages by Length
    metrics = ["Loss Recall", "Payout Precision",'MSE']
    rdf_avg = rdf.groupby("Length")[metrics].mean().reset_index()

    # Normalize each metric (0–1 range)
    rdf_norm = rdf_avg.copy()
    rdf_norm[metrics] = (rdf_avg[metrics] - rdf_avg[metrics].min()) / (rdf_avg[metrics].max() - rdf_avg[metrics].min())

    # Melt to long format
    rdf_long = rdf_norm.melt(id_vars="Length", 
                            value_vars=metrics, 
                            var_name="Metric", 
                            value_name="Normalized Value")

    # Plot
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=rdf_long,
        x="Length",
        y="Normalized Value",
        hue="Metric",
        marker="o",
        lw=2.5,
        palette="Set2"
    )

    plt.title("Normalized Model Performance vs. Training Data Length", fontsize=16, pad=15)
    plt.xlabel("Training Data Length")
    plt.ylabel("Normalized Metric (0–1)")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()