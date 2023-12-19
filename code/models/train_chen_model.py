import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch, time
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive',force_remount=True)

DATA_DIR = '/content/drive/MyDrive/Illinois'

class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()
        self.layers = [nn.Linear(input_size,layer_sizes[0]), nn.ReLU(), nn.Dropout(p=0.01)]
        for i in range(len(layer_sizes)-1):
            hidden_layer = [nn.Linear(layer_sizes[i],layer_sizes[i+1]), nn.ReLU(),  nn.Dropout(p=0.01)]
            self.layers += hidden_layer

        self.layers += [nn.Linear(layer_sizes[-1],1),nn.ReLU()]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        u = self.layers(x)
        return u

def create_chen_loss(fn_args):
    risk_load, w_0, subsidy = fn_args['risk_load'], fn_args['w_0'], fn_args['subsidy']
    premium_ub, premium_lb = fn_args['premium_ub'], fn_args['premium_lb']
    alpha, phi = fn_args['alpha'], fn_args['phi']
    def chen_loss(y_true, y_pred):
        premium = risk_load*y_pred.mean()
        wealth = w_0 - y_true + y_pred - subsidy*premium
        utility = (-1/alpha)*torch.exp(-alpha*wealth)
        zero = torch.tensor([0]).cuda()
        ub_penalty = torch.max(premium-premium_ub,zero)
        lb_penalty = torch.max(premium_lb-premium,zero)
        return -utility.mean() + phi*(torch.pow(ub_penalty,2) + phi*torch.pow(lb_penalty,2))

    return chen_loss

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

def train_model(param_dict,pretrained_model=None):
    param_dict = {'epochs':500,'model_layers':[64,64,16],'risk_load':1.3,'w_0':388.6,'subsidy':1,
                'alpha':0.008,'phi':1e-6,'premium_ub':500,'premium_lb':0}

    epochs = param_dict['epochs']
    epoch_cycle = int(epochs/10)
    model_layers = param_dict['model_layers']

    # load data, standardize, split into train/val/test
    train_X = np.load(os.path.join(DATA_DIR,'train_X.npy')).astype('float32')
    train_y = np.load(os.path.join(DATA_DIR,'train_y.npy')).astype('float32')
    train_X, train_y = remove_nans(train_X,train_y)
    val_X = np.load(os.path.join(DATA_DIR,'val_X.npy')).astype('float32')
    val_y = np.load(os.path.join(DATA_DIR,'val_y.npy')).astype('float32')
    val_X, val_y = remove_nans(val_X,val_y)

    input_size = train_X.shape[1]
    if pretrained_model is not None:
        model = pretrained_model
    else:
        model = MLP(input_size,model_layers).cuda()
    loss_fn = create_chen_loss(param_dict)
    optimizer = torch.optim.RMSprop(model.parameters(),alpha=0.9,lr=0.001)

    train_dataloader = DataLoader(list(zip(train_X,train_y)),batch_size=8000)
    eval_dataloader = DataLoader(list(zip(val_X,val_y)),batch_size=8000)

    # define saving infrastructure
    best_eval_loss = np.inf
    best_epoch = 0
    best_model_state = None
    start = time.time()
    for epoch in range(1,epochs+1):
        train_loss = train_epoch(model, loss_fn, train_dataloader, optimizer)

        if epoch % epoch_cycle == 0:
            eval_loss = validation_loss(model, loss_fn, eval_dataloader)
            print("Epoch {}/{}: train loss: {:.2f}  | Eval loss: {:.2f}".format(epoch, epochs, train_loss, eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state = model.state_dict()
                best_epoch = epoch

    best_model = MLP(input_size,model_layers).cuda()
    best_model.load_state_dict(best_model_state)

    # Add code to test best model and to save results. 
    return best_model

def train_epoch(model, loss_fn, dataloader, optimizer):
    model.train()
    losses = []
    for X, y in dataloader:
        X, y = X.cuda(), y.cuda()
        y_pred = model(X).squeeze()
        layer_weights = [layer.weight for layer in model.layers if isinstance(layer,nn.Linear)]
        norms = [torch.norm(weight,2) for weight in layer_weights]
        loss = loss_fn(y,y_pred).mean() + 0.0001*torch.tensor(norms).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)

def remove_nans(X,y):
    mask = np.isfinite(y)
    if len(X.shape) == 3:
        mask = mask.all(axis=(1,2))
        X = X[mask,:,:]
    else:
        # mask = mask.all(axis=1)
        X = X[mask,:]
    y = y[mask]
    return X, y

def calculate_premium(model, X, risk_load):
    model.eval()
    with torch.inference_mode():
        y_pred = model(X).squeeze()
        y_pred = y_pred.cpu()

    premium = risk_load*y_pred.mean()
    return premium
        
def train_constrained_model(param_dict):
    train_X = np.load(os.path.join(DATA_DIR,'train_X.npy')).astype('float32')
    train_y = np.load(os.path.join(DATA_DIR,'train_y.npy')).astype('float32')
    train_X, train_y = remove_nans(train_X,train_y)

    model = train_model(param_dict)
    premium = calculate_premium(model,train_X,param_dict['risk_load'])
    premium_ub, premium_lb, phi = param_dict['premium_ub'], param_dict['premium_lb'], param_dict['phi']
    
    previous_premium = premium - 1
    while(((premium > premium_ub) or (premium < premium_lb) or (abs((premium-previous_premium)/previous_premium)>0.01)) and (phi < 0.1)):
        previous_premium = premium
        phi = phi*5
        param_dict['phi'] = phi
        model = train_model(param_dict, model)
        premium = calculate_premium(model, train_X, param_dict['risk_load'])
        print(f"Intermediate premium: {premium}")


param_dict = {'epochs':20,'model_layers':[64,16],'risk_load':1.3,'w_0':388.6,'subsidy':1,
              'alpha':0.008,'phi':1e-6,'premium_ub':500,'premium_lb':0}