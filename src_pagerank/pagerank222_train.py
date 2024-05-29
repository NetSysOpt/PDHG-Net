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
from torch.utils.data import TensorDataset, DataLoader
log_file_path = "/model/project_log_new_500.log"
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

def process(model, sample, optimizer, device):
    sample["con_feat"] = sample["con_feat"].to(device)
    sample["edge_index"] = sample["edge_index"].to(device)
    sample["edge_weight"] = sample["edge_weight"].to(device)
    sample["var_feat"] = sample["var_feat"].to(device)
    sample["label"] = sample["label"].to(device)
    sample["b"] = sample["b"].to(device)
    sample["c"] = sample["c"].to(device)
    sample["dual"] = sample["dual"].to(device)

    # Set the model to training mode
    model.train()
    MSE = torch.nn.MSELoss()
    # Forward pass
    primal,dual = model(sample)
    
    if len(torch.where(sample["dual"].unsqueeze(1)==float('-inf'))[0]) >0 or len(torch.where(sample["dual"].unsqueeze(1)==float('inf'))[0]) > 0 :
                sample["dual"] = torch.zeros_like(sample["dual"]).to(device)
    if len(torch.where(sample["dual"].unsqueeze(1)==float('-inf'))[0]) >0 or len(torch.where(sample["label"].unsqueeze(1)==float('inf'))[0]) > 0:
                sample["label"] = torch.zeros_like(sample["label"]).to(device)
    loss1 = MSE(primal, sample["label"].unsqueeze(1))
    loss2 = MSE(dual, sample["dual"].unsqueeze(1))
    loss = loss1 + loss2
    
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
    
    return return_loss

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--ins", default = 190)
parser.add_argument("--gpu", help="gpu index", default="1")
parser.add_argument("--embSize", help="embedding size of GNN", default="64")
parser.add_argument("--epoch", help="num of epoch", default="1")

args = parser.parse_args()

## SET-UP HYPER PARAMETERS
max_epochs = int(args.epoch)

## SET-UP DATASET
dataset = []
for sry in range(args.ins):
    savepath = f'../data/pagerank{sry}.pkl'
    subset = torch.load(savepath, map_location="cpu")
    dataset+=subset
n_Samples = 12345678
m = dataset[0]["b"].shape[0]
n = dataset[0]["c"].shape[0]
nnz = dataset[0]["edge_index"].shape[1]
nConsF = dataset[0]["con_feat"].shape[1]
nVarF = dataset[0]["var_feat"].shape[1]
lr = 0.0001
## SET-UP MODEL
embSize = int(args.embSize)
embSize = int(args.embSize)
if not os.path.exists('../models/'):
    os.mkdir('../models/')
model_path = '../models/'+ '_d' + str(n_Samples) + '_s' + str(embSize) + '.pkl'
set_seed(0)
gpu_index = int(args.gpu)
torch.cuda.set_device(gpu_index)
current_device = torch.cuda.current_device()
print(f"Using GPU {current_device}")
torch.cuda.empty_cache()
print("torch.cuda.is_available():{}".format(torch.cuda.is_available()))

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# print(embSize, nConsF, nVarF)
# quit()
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
print(dataset)
# Main training loop
while epoch <= max_epochs:
    for sample_idx in range(len(dataset)-1):
        print(f"EPOCH: {epoch}!!!\n")
        current_loss = process(model, dataset[sample_idx], optimizer, device)
        print(f"EPOCH: {epoch}, SAMPLE: {sample_idx}, TRAIN LOSS: {current_loss}")
        logging.info(f"EPOCH: {epoch}, SAMPLE: {sample_idx}, TRAIN LOSS: {current_loss}")
        if current_loss < loss_best:
            torch.save(model.state_dict(), model_path)
            print("model saved to:", model_path)
            loss_best = current_loss
    epoch += 1
