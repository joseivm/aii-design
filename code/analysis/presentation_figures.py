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
TABLES_DIR = PROJECT_DIR + '/output/tables'
corr_filename = TABLES_DIR + '/Exploration/correlation_exploration.csv'
eps_filename = TABLES_DIR + '/Exploration/epsilon_exploration.csv'

# Output files/dirs
FIGURES_DIR = PROJECT_DIR + '/output/figures/Presentation'


def correlation_figure():
    cdf = pd.read_csv(corr_filename)
    cdf['a'] = 0.5*(cdf['a_1']+cdf['a_2'])
    cdf['b'] = 0.5*(cdf['b_1']+cdf['b_2'])
    mask = cdf.Correlation >= - 0.85
    plt.close()
    plt.plot(cdf[mask].Correlation, cdf[mask].a,'bo')
    plt.xlabel('Correlation')
    plt.ylabel('Slope of Payout Function')
    plt.show()

    mask = cdf.Correlation >= - 0.85
    plt.close()
    plt.plot(cdf[mask].Correlation, cdf[mask].b,'ro')
    plt.xlabel('Correlation')
    plt.ylabel('Intercept of Payout Function')
    plt.show()

def eps_figure():
    edf = pd.read_csv(eps_filename)
    edf['a'] = 0.5*(edf['a_1']+edf['a_2'])
    edf['b'] = 0.5*(edf['b_1']+edf['b_2'])
    plt.close()
    eps_vals = [0.4,0.17,0.05]
    for idx, row in edf.loc[edf.Epsilon.isin(eps_vals),:].iterrows():
        a = row['a']
        b = row['b']
        eps = row['Epsilon']
        x = np.linspace(4,6,50)
        y = a*x +b
        y = np.maximum(y,0)
        plt.plot(x,y,label=eps,c=plt.cm.Reds(1-eps))
        plt.legend(title='Epsilon')

    plt.xlabel('Predicted Loss')
    plt.ylabel('Payout')
    plt.show()