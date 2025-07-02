import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

RESULTS_DIR = os.path.join(PROJECT_DIR, 'experiments','evaluation','Thailand')

def load_single_zone_results():
    dirpath = os.path.join(RESULTS_DIR, 'Test', 'single-zone results')
    files = [f for f in os.listdir(dirpath) if '.csv' in f]
    rparams = [f.split('_') for f in files]
    fpaths = [os.path.join(dirpath, f) for f in files]
    rdfs = []
    for params, fpath in zip(rparams,fpaths):
        rdf = pd.read_csv(fpath)
        c_k, alpha, w0, _ = params
        rdf['c_k'] = c_k
        rdf['alpha'] = alpha
        rdfs.append(rdf)

    rdf = pd.concat(rdfs,ignore_index=True)
    # 1.  Explicit lookup tables
    ck_map = {
        "ck004": 0.04,
        "ck005": 0.05,
        "ck006": 0.06,
        "ck0":   0.00,
        "ck002": 0.02,
    }

    alpha_map = {
        "r11": 1.1,
        "r15": 1.5,
        "r2":  2.0,
        "r25": 2.5,
        "r3":  3.0,
        "r35": 3.5,
    }

    # 2.  Apply the mapping
    rdf["c_k"]   = rdf["c_k"].map(ck_map)
    rdf["alpha"] = rdf["alpha"].map(alpha_map)
    return rdf

    
