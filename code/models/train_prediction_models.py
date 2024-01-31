import pandas as pd
import os
import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sktime.regression.interval_based import TimeSeriesForestRegressor
import time
import random
import string
from joblib import dump, load
from pathlib import Path


from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
TRANSFORMS_DIR = os.path.join(PROJECT_DIR,'data','time-series-transforms')

# Output files/dirs
RESULTS_DIR = os.path.join(PROJECT_DIR,'experiments','prediction')
MODELS_DIR = os.path.join(PROJECT_DIR,'models')

def load_data(state, transform,length):
    length = str(length)
    X_train_fname = f"X_train_{transform}.npy"
    X_train_full_path = os.path.join(TRANSFORMS_DIR, state,length, X_train_fname)

    X_val_fname = f"X_val_{transform}.npy"
    X_val_full_path = os.path.join(TRANSFORMS_DIR, state,length,X_val_fname)

    X_test_fname = f"X_test_{transform}.npy"
    X_test_full_path = os.path.join(TRANSFORMS_DIR, state,length,X_test_fname)

    y_train_fname = f"y_train_{transform}.npy"
    y_train_full_path = os.path.join(TRANSFORMS_DIR, state,length,y_train_fname)

    y_val_fname = f"y_val_{transform}.npy"
    y_val_full_path = os.path.join(TRANSFORMS_DIR, state,length,y_val_fname)

    y_test_fname = f"y_test_{transform}.npy"
    y_test_full_path = os.path.join(TRANSFORMS_DIR, state,length,y_test_fname)

    X_train = np.load(X_train_full_path)
    X_val = np.load(X_val_full_path)
    X_test = np.load(X_test_full_path)

    y_train = np.load(y_train_full_path)
    y_val = np.load(y_val_full_path)
    y_test = np.load(y_test_full_path)

    cols = ~np.isnan(X_train).any(axis=0)
    # X_train = np.concatenate((X_train, X_val))
    # y_train = np.concatenate((y_train, y_val))

    return X_train[:,cols], y_train, X_val[:,cols], y_val, X_test[:,cols], y_test
    # return {'X_train': X_train[:,cols], 'y_train': y_train, 
    #         'X_test': X_test[:,cols],'y_test': y_test}
    
def train_model(state, transform, algorithm, algorithm_name, alg_args=None, length=98):
    length = str(length)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(state, transform,length)

    start = time.time()
    if alg_args is not None:
        reg = algorithm(**alg_args)
    else:
        reg = algorithm()

    reg.fit(X_train,y_train)

    train_preds = reg.predict(X_train)
    val_preds = reg.predict(X_val)
    test_preds = reg.predict(X_test)
    end = time.time()
    runtime = (end-start)/60

    results = coverage_metrics(y_val, val_preds,0)
    model_name = create_model_name(transform, algorithm_name)
    results['Algorithm'] = algorithm_name
    results['Transform'] = transform
    results['Time'] = np.round(runtime,2)
    results['Model Name'] = model_name
    results['Algorithm Args'] = str(alg_args)
    print(results)
    outdir = os.path.join(RESULTS_DIR,state)
    Path(outdir).mkdir(exist_ok=True)
    save_results(results,outdir,length)

    train_pred_df = pd.DataFrame({'Loss': y_train, 'PredLoss': train_preds, 'Set': 'Train'})
    val_pred_df = pd.DataFrame({'Loss': y_val, 'PredLoss': val_preds, 'Set':'Val'})
    test_pred_df = pd.DataFrame({'Loss': y_test, 'PredLoss': test_preds, 'Set': 'Test'})
    pred_df = pd.concat([train_pred_df, val_pred_df, test_pred_df], ignore_index=True)

    pred_dir = os.path.join(RESULTS_DIR,state,f"predictions {length}")
    Path(pred_dir).mkdir(exist_ok=True)
    pred_filename = os.path.join(pred_dir,f"{model_name}_preds.csv")
    pred_df.to_csv(pred_filename,index=False)

    # model_path = os.path.join(MODELS_DIR,model_name+'.joblib')
    # dump(reg, model_path)
    
def coverage_metrics(y_true, y_pred, strike_pct):
    max_payout = 1500
    strike_val = strike_pct*max_payout
    eval_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    eval_df['Ideal Payout'] = np.maximum(eval_df['y_true'] - strike_val, 0)
    eval_df['Payout'] = np.minimum(np.maximum(eval_df['y_pred'] - strike_val, 0),max_payout)
    eval_df['Max Payout'] = np.minimum(eval_df['Payout'], eval_df['Ideal Payout'])

    eval_df['Contract Coverage'] = eval_df['y_true']-strike_val
    eval_df['Total Loss Recall'] = eval_df['Max Payout']/eval_df['y_true']
    eval_df['Covered Loss Recall'] = eval_df['Max Payout']/eval_df['Ideal Payout']
    eval_df['Payout Precision'] = eval_df['Max Payout']/eval_df['Payout']

    covered_recall = eval_df.loc[eval_df['y_true'] > strike_val,'Covered Loss Recall'].mean()
    total_recall = eval_df.loc[eval_df['y_true'] > 0, 'Total Loss Recall'].mean()
    payout_precision = eval_df.loc[eval_df['y_pred'] > strike_val, 'Payout Precision'].mean()
    cost = eval_df['Payout'].mean()
    cost_per_coverage = cost/total_recall
    metrics_dict = {
        'Loss Recall': total_recall,
        'Payout Precision': payout_precision,
        'Average Cost': cost,
        'Cost Per Coverage': cost_per_coverage,
        'MSE': metrics.mean_squared_error(y_true,y_pred)
    }
    return metrics_dict

def save_results(metrics_dict, results_dir, length):
    mdf = pd.DataFrame([metrics_dict])
    results_file = os.path.join(results_dir, f"results_{length}.csv")
    if os.path.isfile(results_file):
        rdf = pd.read_csv(results_file)
    else: 
        rdf = pd.DataFrame()
    rdf = pd.concat([rdf,mdf],ignore_index=True)
    rdf.to_csv(results_file,float_format = '%.3f',index=False)

def create_model_name(transform, algorithm):
    model_name = f"{transform}_{algorithm}_"
    model_id = ''.join(random.choices(string.ascii_letters + string.digits,k=3))

    return model_name + model_id

lengths = [i*10 for i in range(3,9)] + [83]
# states = ['Illinois','Iowa','Missouri','Indiana']
states = ['Missouri']
transforms = ['chen','catch22','rocket']
algorithms = {'Ridge': (Ridge,None), 'Lasso': (Lasso,None), 'SVR': (SVR,None), 
              'Random Forest': (RandomForestRegressor,{'n_estimators':250}),
              'Gradient Boosting':(GradientBoostingRegressor,{'n_estimators':250})}

# train_model('Iowa','rocket',GradientBoostingRegressor,'Gradient Boosting',{'n_estimators':250},83)

start = time.time()
for state in states:
    for length in lengths:
        for transform in transforms:
            print(f"###### {state} ### {length} ### {transform}#####")
            for name, (algorithm,alg_args) in algorithms.items():
                if not (name == 'Random Forest' and transform == 'rocket'):
                    train_model(state, transform, algorithm, name, alg_args, length)
end = time.time()
runtime = (end-start)/60
print(f"Runtime: {runtime}")