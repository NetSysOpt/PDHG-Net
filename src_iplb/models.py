import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch import Tensor
from torch.nn import Parameter
import os
import psutil
from torch.nn import Parameter
import torch.sparse as sp
import pynvml
#23,23,25,27,28,29,40,42,44,45,46,47,50,56,57,59,60,62,66,67,70,77,80,82,85,95,114,

def get_gpu_mem_info(gpu_id=0):

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used





# def ruiz_normalize(matrix):
#     norms = torch.sqrt(torch.sum(matrix ** 2, dim=0) + 1e-8)
#     normalized_matrix = matrix / norms
#     return normalized_matrix


def build_adj_left_from_right(edge_index: Tensor, edge_weight: Tensor, n_node_left: int, n_node_right: int):
    # Create a sparse matrix in COO (Coordinate) format
    indices = edge_index.t()
    values = edge_weight
    shape = (n_node_left, n_node_right)

    # Use torch.sparse_coo_tensor instead of sp.FloatTensor
    adj_coo = torch.sparse_coo_tensor(indices, values, torch.Size(shape))

    # Convert COO sparse matrix to CSR (Compressed Sparse Row) format
    adj_csr = adj_coo.coalesce()
    #     print(dir(adj_csr._values()))
    #     print(adj_csr._values().sum(dim=1).shape)
   # Normalize rows of the adjacency matrix in GCN style
    row_sum = adj_csr._values().sum(dim=-1)
#     row_sum = torch.max(adj_csr._values())
    row_sum = torch.clamp(row_sum, min=1)  # Avoid division by zero
#     row_sum = torch.sqrt(row_sum)
    row_norm = 1.0 / row_sum
    row_norm = row_norm.unsqueeze(-1)

    # Reconstruct sparse matrix after normalization
    adj_csr = torch.sparse_coo_tensor(adj_csr._indices(), adj_csr._values() * row_norm, adj_csr.shape)

    # Move the sparse matrix to GPU if edge_index is on GPU
    adj_csr = adj_csr.cuda() if edge_index.is_cuda else adj_csr

    return adj_csr, row_norm


def build_adj_right_from_left(edge_index: Tensor, edge_weight: Tensor, n_node_left: int, n_node_right: int):
    # Create a sparse matrix in COO (Coordinate) format
    indices = edge_index.t()
    values = edge_weight
    shape = (n_node_right, n_node_left)

    # Use torch.sparse_coo_tensor instead of sp.FloatTensor
    adj_coo = torch.sparse_coo_tensor(indices, values, torch.Size(shape))

    # Convert COO sparse matrix to CSR (Compressed Sparse Row) format
    adj_csr = adj_coo.coalesce()

    # Normalize rows of the adjacency matrix in GCN style
    row_sum = adj_csr._values().sum(dim=-1)
#     row_sum = torch.max(adj_csr._values())
    row_sum = torch.clamp(row_sum, min=1)  # Avoid division by zero
#     row_sum = torch.sqrt(row_sum)
    row_norm = 1.0 / row_sum
    row_norm = row_norm.unsqueeze(-1)

    # Reconstruct sparse matrix after normalization
    adj_csr = torch.sparse_coo_tensor(adj_csr._indices(), adj_csr._values() * row_norm, adj_csr.shape)

    # Move the sparse matrix to GPU if edge_index is on GPU
    adj_csr = adj_csr.cuda() if edge_index.is_cuda else adj_csr

    return adj_csr, row_norm


class BipartiteGraphConvolution(nn.Module):
    def __init__(self, emb_size, activation, initializer, right_to_left=False, **kwargs):
        super(BipartiteGraphConvolution, self).__init__(**kwargs)
        self.adj_left_from_right = None
        self.adj_right_from_left = None
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.right_to_left = right_to_left
        self.conv1 = GCNConv(self.emb_size, self.emb_size)
        self.output_module = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=True)
        )
        self.temp = Parameter(torch.Tensor(2))
        self.activation_temp = nn.Sigmoid()
        ## new revision
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(0.9)
        #super(BipartiteGraphConvolution, self).reset_parameters()

    def forward(self, left_features, right_features_k, edge_index, edge_weight, right_features, c, b):
        n = c.shape[0]
        m = b.shape[0]
        if self.right_to_left:
            self.adj_left_from_right, rn = build_adj_left_from_right(edge_index, edge_weight, m, n)
            conv_output = torch.matmul(self.adj_left_from_right, (2 * right_features - right_features_k))
            output = self.output_module(left_features - self.temp[0] * (b - conv_output))
        else:
            self.adj_right_from_left, rn = build_adj_right_from_left(edge_index, edge_weight, m, n)
            conv_output = torch.matmul(self.adj_right_from_left, left_features)
            output = self.output_module(right_features + self.temp[1] * (c - conv_output))
        return output


class GCNPolicy(nn.Module):
    def __init__(self, emb_size, nConsF, nVarF, covolution_num=4,isGraphLevel=True):
        super(GCNPolicy, self).__init__()

        self.emb_size = emb_size
        self.cons_nfeats = nConsF
        self.var_nfeats = nVarF
        self.is_graph_level = isGraphLevel

        self.activation = nn.ReLU()
        #         self.activation_final = nn.Sigmoid()
        self.cons_embedding = nn.Sequential(
            nn.Linear(in_features=self.cons_nfeats, out_features=self.emb_size, bias=True),
            nn.ReLU()
        )

        self.edge_embedding = nn.Sequential()

        self.var_embedding = nn.Sequential(
            nn.Linear(in_features=self.var_nfeats, out_features=self.emb_size, bias=True),
            nn.ReLU()
        )

        ## convolution defination
        self.convolution_num = covolution_num
        self.conv_v_to_c = torch.nn.ModuleList()
        self.conv_c_to_v = torch.nn.ModuleList()

        for _ in range(covolution_num):
            self.conv_v_to_c.append(
                BipartiteGraphConvolution(self.emb_size, self.activation, None, right_to_left=True, ))
            self.conv_c_to_v.append(BipartiteGraphConvolution(self.emb_size, self.activation, None))
        # self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, self.activation, None, right_to_left=True, ),

        # self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, self.activation, None)

        self.output_module1 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.emb_size, out_features=1, bias=False)
        )
        self.output_module2 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.emb_size, out_features=1, bias=False)
        )

    
    
    def forward(self, constraint_features, edge_indices, edge_features, variable_features, c, b):
        final_features = 0

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        for count in range(self.convolution_num):
            variable_features_k = variable_features
            variable_features = self.conv_c_to_v[count](
                constraint_features, variable_features_k, edge_indices, edge_features, variable_features, c, b)
            variable_features = self.activation(variable_features)

            constraint_features = self.conv_v_to_c[count](
                constraint_features, variable_features_k, edge_indices, edge_features, variable_features, c, b)
            constraint_features = self.activation(constraint_features)

        output1 = self.output_module1(variable_features)
        output2 = self.output_module2(constraint_features)
        return output1, output2

    def save_state(self, path):
        state_dict = {}
        for name, param in self.state_dict().items():
            state_dict[name] = param.cpu().numpy()
        torch.save(state_dict, path)

    def restore_state(self, path):
        state_dict = torch.load(path)
        for name, param in self.state_dict().items():
            param.copy_(torch.from_numpy(state_dict[name]))

    def reset_parameters(self):
        super(GCNPolicy, self).reset_parameters()
        self.conv_v_to_c.reset_parameters()
        self.conv_c_to_v.reset_parameters()
