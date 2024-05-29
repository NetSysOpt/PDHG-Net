import numpy as np
from pandas import read_csv
import argparse
import os
from models import GCNPolicy
import torch
import random
import torch.nn as nn
import torch.optim as optim
import logging
import gzip
import pickle
from torch.utils.data import TensorDataset, DataLoader
log_file_path = "../records/lb_training.log"
with open(log_file_path, 'a'):
    pass
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("500*100000")
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def is_empty_tensor(tensor):
    return torch.is_tensor(tensor) and torch.numel(tensor) == 0
def process(model, sample, optimizer, device, training=True):
    sample["con_feat"] = sample["con_feat"].to(device)
    sample["edge_index"] = sample["edge_index"].to(device)
    sample["edge_weight"] = sample["edge_weight"].to(device)
    sample["var_feat"] = sample["var_feat"].to(device)
    sample["label"][torch.where(sample["label"]>=1e4)]=0
    sample["label"][torch.where(sample["label"]<=-1e4)]=0
    sample["label"] = sample["label"].to(device)
    sample["b"] = sample["b"].to(device)
    sample["c"] = sample["c"].to(device)
    sample["dual"][torch.where(sample["dual"]>=1e4)]=0
    sample["dual"][torch.where(sample["dual"]<=-1e4)]=0
    sample["dual"] = sample["dual"].to(device)
    model.train()
    MSE = torch.nn.MSELoss()
    primal,dual = model(sample["con_feat"], sample["edge_index"], sample["edge_weight"], sample["var_feat"], sample["c"], sample["b"])

    if len(torch.where(sample["dual"].unsqueeze(1)==float('-inf'))[0]) >0 or len(torch.where(sample["dual"].unsqueeze(1)==float('inf'))[0]) > 0 :
        print("senmiao")
        sample["dual"] = torch.zeros_like(sample["dual"]).to(device)
    if len(torch.where(sample["label"].unsqueeze(1)==float('-inf'))[0]) >0 or len(torch.where(sample["label"].unsqueeze(1)==float('inf'))[0]) > 0:
        print("ruoyu")
        sample["label"] = torch.zeros_like(sample["label"]).to(device)
    if is_empty_tensor(sample["label"]):
        sample["label"] = torch.zeros_like(primal.squeeze())
    if is_empty_tensor(sample["dual"]):
        sample["dual"] = torch.zeros_like(dual.squeeze())
    loss1 = MSE(primal, sample["label"].unsqueeze(1))
    loss2 = MSE(dual, sample["dual"].unsqueeze(1))
    loss = loss1 + loss2

    # Backward pass
    if training:
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
    del sample
    
    return return_loss

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
# parser.add_argument("--data", help="number of training data", default=50)
parser.add_argument("--ins", default = 1000)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default="64")
parser.add_argument("--epoch", help="num of epoch", default="100")
parser.add_argument("-i","--ident", type=str, default="")

args = parser.parse_args()

## SET-UP HYPER PARAMETERS
max_epochs = int(args.epoch)

## SET-UP DATASET
dataset = []
savepath = f'../data/lb/lp_1.pkl'
fgz= gzip.open(savepath,'rb') 
subset = pickle.load(fgz) 
fgz.close()
# subset = torch.load(savepath, map_location="cpu")
dataset+=subset
# n_Samples = 'ip1e5_1000_new'
# n_Samples = 'lb1e5_1000_new_1000'
#lb1e5_1000_new_1000_epoch_1
# n_Samples = 'lb1e5_1000_new_1000_epoch_1_extra'
n_Samples = 'lb1e5_1000_new_1000'
#lb_small_1e5_1000_new_1000
n_Samples = args.ident
m = dataset[0]["b"].shape[0]
n = dataset[0]["c"].shape[0]
nnz = dataset[0]["edge_index"].shape[1]
nConsF = dataset[0]["con_feat"].shape[1]
nVarF = dataset[0]["var_feat"].shape[1]
lr = 0.0001
## SET-UP MODEL
embSize = int(args.embSize)
model_path = '../models/lb_small_' + n_Samples + '_s' + str(embSize) + '.pkl'
set_seed(0)
gpu_index = int(args.gpu)
torch.cuda.set_device(gpu_index)
current_device = torch.cuda.current_device()
print(f"Using GPU {current_device}")
torch.cuda.empty_cache()
print("torch.cuda.is_available():{}".format(torch.cuda.is_available()))

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
model = GCNPolicy(embSize, nConsF, nVarF, isGraphLevel=False)
z = 0
for name, i in model.named_parameters():
    if "weight" in name and len(i.size())>1:
        z = z + 1
        torch.nn.init.xavier_uniform_(i)
print("Xavier initialization done with " + str(z) + " weights")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
epoch = 0
count_restart = 0
err_best = 2
loss_best = float('inf')
# Main training loop

n_train = 950
file_nums=[]
for i in range(1000):
    file_nums.append(i)
training_set = random.sample(file_nums,n_train)
valid_set = [x for x in file_nums if x not in training_set]

while epoch <= max_epochs:
    total_sample = 0
    
    random.shuffle(training_set)
    total_train_loss = 0
    for v_indx in range(n_train):
        try:
            sample_idx=training_set[v_indx]
            savepath = f'../data/lb_small/lb_small/lp_{int(sample_idx)}.pkl'
            fgz= gzip.open(savepath,'rb') 
            sample = pickle.load(fgz) 
            fgz.close()
        except:
            print("error of f")
            continue
        for instance_idx in range(len(sample)):
            total_sample +=1
            current_loss = process(model, sample[instance_idx], optimizer, device)
            total_train_loss+=current_loss
            print(f"EPOCH: {epoch}, total sample: {total_sample}, average train loss: {total_train_loss/total_sample}   | This loss: {current_loss}")
            logging.info(f"EPOCH: {epoch}, total sample: {total_sample}, average train loss: {total_train_loss/total_sample}   | This loss: {current_loss}")
                
                
    print(f'EPOCH: {epoch} VALIDATION started') 
    total_sample=0
    total_val_loss = 0
    for v_indx in range(1000-n_train):
        sample_idx=valid_set[v_indx]
        savepath = f'../data/lb_small/lb_small/lp_{int(sample_idx)}.pkl'
        fgz= gzip.open(savepath,'rb') 
        sample = pickle.load(fgz) 
        fgz.close()
        for instance_idx in range(len(sample)):
            total_sample +=1
            current_loss = process(model, sample[instance_idx], optimizer, device, False)
            total_val_loss += current_loss
            print(f"Validation EPOCH: {epoch}, total sample: {total_sample}, average validation loss: {total_val_loss/total_sample}   | This loss: {current_loss}")
            logging.info(f"Validation EPOCH: {epoch}, total sample: {total_sample}, average validation loss: {total_val_loss/total_sample}   | This loss: {current_loss}")
    if total_val_loss < loss_best:
        torch.save(model.state_dict(), model_path)
        print("model saved to:", model_path)
        loss_best = current_loss
    
    
    logging.info(f"运行到第{epoch}个epoch了")
    epoch += 1
