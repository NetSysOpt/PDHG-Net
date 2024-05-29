import numpy as np
from pandas import read_csv
import argparse
import os

import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process(model, sample, optimizer, device):
    sample["con_feat"] = sample["con_feat"].to(device)
    sample["edge_index"] = sample["edge_index"].to(device)
    sample["edge_weight"] = sample["edge_weight"].to(device)
    sample["var_feat"] = sample["var_feat"].to(device)
    sample["label"] = sample["label"].to(device)
    sample["b"] = sample["b"].to(device)
    sample["c"] = sample["c"].to(device)
    sample["dual"] = sample["dual"].to(device)
    #print("con shape", sample["con_feat"].shape)
    #print("vat shape", sample["var_feat"].shape)
    #print("label", sample["label"].shape)
    # Set the model to training mode
    model.train()
    ## MSE
    MSE = torch.nn.MSELoss()
    # Forward pass
    primal,dual = model(sample)
    #print(primal.shape)
    loss1 = MSE(primal, sample["label"].unsqueeze(1))
    loss2 = MSE(dual, sample["dual"].unsqueeze(1))
    loss = loss1 + loss2
#     loss = torch.linalg.norm(logits-sample["label"].unsqueeze(1),ord=np.inf)
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return_loss = loss.item()
    
    sample["con_feat"] = sample["con_feat"].cpu()
    sample["edge_index"] = sample["edge_index"].cpu()
    sample["edge_weight"] = sample["edge_weight"].cpu()
    sample["var_feat"] = sample["var_feat"].cpu()
    sample["label"] = sample["label"].cpu()
    sample["b"] = sample["b"].cpu()
    sample["c"] = sample["c"].cpu()
    sample["dual"] = sample["dual"].cpu()
    
    #del sample
  
    return return_loss

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
# parser.add_argument("--data", help="number of training data", default=50)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default="64")
parser.add_argument("--epoch", help="num of epoch", default="20")
parser.add_argument("--depth", help="num of depth",type=int, default=2)
parser.add_argument("--model", help="type of model", default="GCN")
parser.add_argument("--n_files", help="n_training_files", default="100")
parser.add_argument("--seed", help="random seed", type=int, default=0)
parser.add_argument("--model_type", help="GCN", default="GCN")

args = parser.parse_args()

if args.model_type == "GCN":
    from models import GCNPolicy
elif args.model_type == "SGC":
    from models_fixlayer import GCNPolicy

## SET-UP HYPER PARAMETERS
max_epochs = int(args.epoch)

## SET-UP DATASET

#dataset = torch.load(savepath, map_location="cpu")

dataset = []
for sry in range(int(float(args.n_files)/50)):
    savepath = f'../data/IPdata{sry}.pkl'
    subset = torch.load(savepath, map_location="cpu")
#     for i in subset:
#         dataset.append(i)
    dataset+=subset

n_Samples = 12345678
m = dataset[0]["b"].shape[0]
n = dataset[0]["c"].shape[0]
nnz = dataset[0]["edge_index"].shape[1]
nConsF = dataset[0]["con_feat"].shape[1]
nVarF = dataset[0]["var_feat"].shape[1]
lr = 0.0001

## SET-UP MODEL
if args.seed!=0:
    seed_str = "_seed" + str(args.seed)
else:
    seed_str = ""

if args.model_type!="GCN":
    model_str = "_model" + args.model_type
else:
    model_str = ""
embSize = int(args.embSize)
model_path = '/code/GNN-LP/models/'+ args.model + '_d'+str(args.depth) + '_w' + str(args.embSize) + '_data' + str(int(args.n_files))+model_str+ seed_str + '.pt'
log_path = '/code/GNN-LP/records/'+"train_loss_"+ args.model + '_d'+str(args.depth) + '_w' + str(args.embSize) + '_data' + str(int(args.n_files))+model_str  + seed_str +'.csv'


gpu_index = int(args.gpu)
torch.cuda.set_device(gpu_index)
current_device = torch.cuda.current_device()
print(f"Using GPU {current_device}")
torch.cuda.empty_cache()
print("torch.cuda.is_available():{}".format(torch.cuda.is_available()))

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)
model = GCNPolicy(embSize, nConsF, nVarF, covolution_num = args.depth, isGraphLevel=False)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
# loss_init = process(model, dataset[0], optimizer, device)
epoch = 0
count_restart = 0
err_best = 2
loss_best = 1e6
epoch_list = []
best_loss_list = []
epoch_loss_list = []

# Main training loop
while epoch <= max_epochs:
    loss_epoch = 0
    for sample_idx in range(len(dataset)):
        print(sample_idx)
#         print(dataset)
        current_loss = process(model, dataset[sample_idx], optimizer, device)
        print(f"EPOCH: {epoch}, SAMPLE: {sample_idx}, TRAIN LOSS: {current_loss}")
        loss_epoch +=current_loss
    loss_epoch = loss_epoch/len(dataset)
      
    if loss_epoch < loss_best:
        torch.save(model.state_dict(), model_path)
        print("model saved to:", model_path)
        loss_best = loss_epoch
    print(f"epoch {epoch} done")
    print(f"current loss: {loss_epoch}    best loss:{loss_best}")
    epoch += 1
    epoch_list.append(epoch)
    best_loss_list.append(loss_best)
    epoch_loss_list.append(loss_epoch)
import pandas as pd
df1 = pd.DataFrame({'epoch':epoch_list,"best_loss": best_loss_list,"epoch_loss":epoch_loss_list})
df1.to_csv(log_path,index=False)
