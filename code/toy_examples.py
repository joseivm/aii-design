import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import cvxpy as cp
import matplotlib.pyplot as plt

# TODO: 1. figure out how to include positive and negative losses as well
# 2. figure out how to set w_bar when w is supposed to be loss

def make_single_zone_data(beta,mu,sigma,n=100):
    theta = np.random.normal(loc=mu,scale=sigma,size=n)
    epsilon = np.random.normal(loc=0,scale=1)
    w = beta*theta + epsilon
    return w, theta.reshape(-1,1)

def min_premium_program(pred_y,train_y,loss_p,epsilon,model='full',a_lb=0):
    w_bar = np.quantile(train_y,loss_p)
    n_samples = train_y.shape[0]
    I = cp.Variable(n_samples)
    a = cp.Variable()
    b = cp.Variable()
    t = cp.Variable()
    gamma = cp.Variable(n_samples)

    constraints = []
    constraints.append(I >= 0)
    constraints.append(t + (1/epsilon)*(1/n_samples)*(sum(gamma))<= w_bar)
    constraints.append(gamma >=  pred_y + (1/n_samples)*sum(I) - I -t)
    constraints.append(gamma >= 0)
    
    if model == 'full':
        constraints.append(I >= a*pred_y + b)
        constraints.append(b <= 0)
    elif model == 'slope':
        constraints.append(I >= a*pred_y)
        constraints.append(a >= a_lb)
    else:
        constraints.append(I >= pred_y + b)
        constraints.append(b <= 0)

    objective = cp.Minimize(sum(I))
    problem = cp.Problem(objective,constraints)
    problem.solve()
    print(problem.status)
    return_dict = {'full':(a.value,b.value),'slope':(a.value,0),'intercept':(1,b.value)}
    return return_dict[model]

def min_CVaR_program(pred_y,train_y,max_pi,epsilon,model='full',a_lb=0):
    n_samples = train_y.shape[0]
    I = cp.Variable(n_samples)
    a = cp.Variable()
    b = cp.Variable()
    t = cp.Variable()
    max_payout = cp.Variable()
    gamma = cp.Variable(n_samples)

    constraints = []
    constraints.append(I >= 0)
    constraints.append(I <= max_payout)
    constraints.append((1/n_samples)*sum(I) <= max_pi*max_payout)
    constraints.append(gamma >=  pred_y + (1/n_samples)*sum(I) - I -t)
    constraints.append(gamma >= 0)
    
    if model == 'full':
        constraints.append(I >= a*pred_y + b)
        constraints.append(b <= 0)
    elif model == 'slope':
        constraints.append(I >= a*pred_y)
        constraints.append(a >= a_lb)
    else:
        constraints.append(I >= pred_y + b)
        constraints.append(b <= 0)

    objective = cp.Minimize(t + (1/epsilon)*(1/n_samples)*(sum(gamma)))
    problem = cp.Problem(objective,constraints)
    problem.solve()
    print(max_payout.value)
    return_dict = {'full':(a.value,b.value),'slope':(a.value,0),'intercept':(1,b.value)}
    return return_dict[model] 

def determine_strike_value(train_y,eval_y,eval_x,pred_model):
    strike_percentiles = np.arange(0.1,0.35,0.05)
    strike_vals = np.quantile(train_y,strike_percentiles)
    
    pred_losses = pred_model.predict(eval_x)
    strike_performance = {}

    for strike_percentile, strike_val in zip(strike_percentiles,strike_vals):
        strike_percentile = np.around(strike_percentile,2)
        insured_loss = np.maximum(eval_y-strike_val,0)
        payout = np.maximum(pred_losses-strike_val,0).reshape(-1,1)
        loss_share_model = LinearRegression().fit(payout,insured_loss)
        share_explained = loss_share_model.coef_[0]
        strike_performance[(strike_percentile,strike_val)] = share_explained

    best_strike_percentile, best_strike_val = max(strike_performance,key=strike_performance.get)
    return best_strike_percentile, best_strike_val

def make_eval_dfs(test_x,test_y,pred_model,strike_val,a,b):
    edf = pd.DataFrame()
    loss_quantiles = np.quantile(test_y,[0.7,0.8,0.9,0.95])
    col_names = ['P(L > {})'.format(i) for i in [70,80,90,95]]
    
    bdf['Losses'] = test_y
    bdf['PredictedLosses'] = pred_model.predict(test_x)
    bdf['Payout'] = np.maximum(bdf['PredictedLosses']-strike_val,0)
    bdf['NetLoss'] = bdf['Losses'] - bdf['Payout']
    
    odf['Losses'] = test_y
    odf['PredictedLosses'] = pred_model.predict(test_x)
    odf['Payout'] = np.maximum(a*odf['PredictedLosses']+b,0)
    odf['NetLoss'] = odf['Losses']-odf['OptPayout']
    
    for col_name, loss_quantile in zip(col_names,loss_quantiles):
        bdf[col_name] = bdf['NetLoss'] > loss_quantile
        odf[col_name] = odf['NetLoss'] > loss_quantile
    return bdf, odf

np.random.seed(1)
train_y, train_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=1000)
eval_y, eval_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=20)
test_y, test_x = make_single_zone_data(beta=3,mu=5,sigma=2,n=1000)

epsilon = 0.01
pred_model = LinearRegression().fit(train_x,train_y)
model_predictions = pred_model.predict(train_x)
strike_p, strike_val = determine_strike_value(train_y,eval_y,eval_x,pred_model)
a,b = optimization_program(model_predictions,train_y,0.75,epsilon)
print('a: {}, b: {}'.format(a,b))


plt.plot(edf['PredictedLosses'],edf['BaselinePayout'],'bs',label='baseline')
plt.plot(edf['PredictedLosses'],edf['OptPayout'],'g^',label='opt')
plt.legend()
plt.show()
plt.close()