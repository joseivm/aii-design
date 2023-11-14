import pandas as pd
import numpy as np
import os, sys

from torch.utils.data import DataLoader
import torch.nn as nn
import torch, time

from code.models.chen_models import MLP

# Input files/dirs
DATA_DIR = os.path.join('data','processed','Illinois')

def create_chen_loss(fn_args):
    risk_load, w_0, subsidy = fn_args['risk_load'], fn_args['w_0'], fn_args['subsidy']
    premium_ub, premium_lb = fn_args['premium_ub'], fn_args['premium_lb']
    alpha, phi = fn_args['alpha'], fn_args['phi']
    def chen_loss(y_true, y_pred):
        premium = risk_load*y_pred.mean()
        wealth = w_0 - y_true + y_pred - subsidy*premium
        utility = (-1/alpha)*torch.exp(-alpha*wealth)
        ub_penalty = torch.maximum(premium-premium_ub,0)
        lb_penalty = torch.maximum(premium_lb-premium,0)
        return utility.mean() + phi*(torch.pow(ub_penalty,2) + torch.pow(lb_penalty,2))
    
    return chen_loss

def chen_loss2(y_true, y_pred, fn_args): 
    risk_load, w_0, subsidy = fn_args['risk_load'], fn_args['w_0'], fn_args['subsidy']
    premium_ub, premium_lb = fn_args['premium_ub'], fn_args['premium_lb']
    alpha, phi = fn_args['alpha'], fn_args['phi']
    premium = risk_load*y_pred.mean()
    wealth = w_0 - y_true + y_pred - subsidy*premium
    utility = (-1/alpha)*torch.exp(-alpha*wealth)
    ub_penalty = torch.maximum(premium-premium_ub,0)
    lb_penalty = torch.maximum(premium_lb-premium,0)
    return utility.mean() + phi*(torch.pow(ub_penalty,2) + torch.pow(lb_penalty,2))

def train_model(param_dict):
    # Model params: model name, model architecture, epochs, 
    epochs = param_dict['epochs']
    epoch_cycle = int(epochs/10)
    model_layers = param_dict['model_layers']

    # load data, standardize, split into train/val/test
    train_X = np.load(os.path.join(DATA_DIR,'train_X.npy'))
    train_y = np.load(os.path.join(DATA_DIR,'train_y.npy'))
    val_X = np.load(os.path.join(DATA_DIR,'val_X.npy'))
    val_y = np.load(os.path.join(DATA_DIR,'val_y.npy'))

    input_size = train_X.shape[1]
    model = MLP(input_size,model_layers)
    loss_fn = create_chen_loss(param_dict)
    optimizer = torch.optim.RMSprop(model.parameters(),alpha=0.9,lr=0.001)

    train_dataloader = DataLoader(list(zip(train_X,train_y)))
    eval_dataloader = DataLoader(list(zip(val_X,val_y)))

    # define saving infrastructure
    best_eval_loss = np.inf
    best_epoch = 0
    best_model_state = None
    start = time.time()
    for epoch in range(1,epochs+1):
        train_loss = train_epoch(model, loss_fn, optimizer)

        if epoch % epoch_cycle == 0:
            eval_loss = validation_loss(model, loss_fn, eval_dataloader)
            print("Epoch {}/{}: train loss: {:.2f}  | Eval loss: {:.2f}".format(epoch, epochs, train_loss, eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state = model.state_dict()
                best_epoch = epoch

    best_model = MLP(input_size,model_layers)
    best_model.load_state_dict(best_model_state)
    test(best_model)
    pass

def validation_loss(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for X,y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X).squeeze()
            loss = loss_fn(y, pred).sum()
            total_loss += loss.item()

    return total_loss



def train_constrained_model(param_dict):
    pass

def train_epoch(model, loss_fn, dataloader, optimizer):
    model.train()
    losses = []
    for X, y in dataloader:
        X, y = X.cuda(), y.cuda()
        y_pred = model(X).squeeze()
        loss = loss_fn(y,y_pred).mean() + torch.norm(model.layer.weight,p=2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return np.mean(losses)
    
