from ortools.linear_solver import pywraplp

from ortools.linear_solver import linear_solver_pb2

from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
from ortools.pdlp.python import pdlp

from ortools.linear_solver.python import model_builder
from ortools.linear_solver.python import model_builder_helper as mbh
import random
import numpy as np
import psutil
import pickle
pcs = psutil.Process()
import networkx as nx
import scipy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import GCNPolicy  # Make sure to import your PyTorch GCNPolicy implementation
import random
import argparse
from ortools.linear_solver.python import model_builder
from ortools.linear_solver.python import model_builder_helper as mbh
import gzip
import logging
import time
from ortools.linear_solver.python import model_builder
from ortools.linear_solver.python import model_builder_helper as mbh
import numpy as np
import networkx as nx
import gzip
import os
# import torch
import math
import time
import torch
import psutil
import pickle
log_file_path = "../records/project_log_new_1000.log"

parser = argparse.ArgumentParser()
# parser.add_argument("--data", help="number of training data", default=50)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default="64")
parser.add_argument("--epoch", help="num of epoch", default="10000")
parser.add_argument("-i","--ident", type=str, default="")
parser.add_argument("-s","--stepsize", type=float, default=0.1)
parser.add_argument("-p","--primalweight", type=float, default=0.1)
parser.add_argument("-n","--ntest", type=int, default=1)
parser.add_argument("-c","--code", type=str, default="code")

args = parser.parse_args()

if 'small' in args.ident:
    from knapsack_ip_small import *
    log_file_path=f'/{args.code}/GNN-LP/logs/ip_small_{args.stepsize}_{args.primalweight}.log'
else:
    from knapsack_ip import *
    log_file_path=f'/{args.code}/GNN-LP/logs/ip_{args.stepsize}_{args.primalweight}.log'




with open(log_file_path, 'a'):
    pass
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("1000*10000")
def solve_get_sol3(model):
    ptc=model.export_to_proto()
    print("hello")
    mdl = get_prob(ptc)
    print("hello")
    params = solvers_pb2.PrimalDualHybridGradientParams()
    opt_criteria = params.termination_criteria.simple_optimality_criteria
#     opt_criteria = set_crit(opt_criteria)
    opt_criteria.eps_optimal_relative = 1.0e-4
    result = pdlp.primal_dual_hybrid_gradient(
        mdl, params
    )
#     for e in dir(mdl):
#         print(e)
    ps=result.primal_solution
    ds=result.dual_solution
    vns=mdl.variable_names
    cns=mdl.constraint_names
    x={}
    y={}
    for idx,v in enumerate(vns):
        x[v]=ps[idx]
    for idx,c in enumerate(cns):
        y[c]=ds[idx]
#     print(ps)
#     print(ds)
    return x,y
def solve_get_sol2(model,sol1,dual1):
    ptc=model.export_to_proto()
    print("hello")
    mdl = get_prob(ptc)
    print("hello")
    params = solvers_pb2.PrimalDualHybridGradientParams()
    opt_criteria = params.termination_criteria.simple_optimality_criteria
    opt_criteria.eps_optimal_relative = 1.0e-4
    
#     opt_criteria.eps_optimal_absolute = 1.0e-6
#     params.l_inf_ruiz_iterations = 0
#     params.l2_norm_rescaling = False
    result = pdlp.primal_dual_hybrid_gradient(
        mdl, params
    )
    for e in dir(mdl):
        print(e)
    ps=result.primal_solution
    ds=result.dual_solution
    vns=mdl.variable_names
    cns=mdl.constraint_names
    x={}
    y={}
    for idx,v in enumerate(vns):
        sol1.append(ps[idx])
    for idx,c in enumerate(cns):
        dual1.append(ds[idx])
    print(ps)
    print(ds)
    return sol1,dual1
def get_prob(prt,realx_int=False,include_name=True):
    return pdlp.qp_from_mpmodel_proto(prt,realx_int,include_name)
def process(model, dataset, device):
    #     for i in range(len(dataset)-1):
    dataset["con_feat"] = dataset["con_feat"].to(device)
    dataset["edge_index"] = dataset["edge_index"].to(device)
    dataset["edge_weight"] = dataset["edge_weight"].to(device)
    dataset["var_feat"] = dataset["var_feat"].to(device)
    dataset["label"] = dataset["label"].to(device)
    dataset["b"] = dataset["b"].to(device)
    dataset["c"] = dataset["c"].to(device)
    dataset["dual"] = dataset["dual"].to(device)
    names=dataset['names']
    
    time11 = time.time()
#     logits, dual = model(dataset)
    logits, dual = model(dataset["con_feat"], dataset["edge_index"], dataset["edge_weight"], dataset["var_feat"], dataset["c"], dataset["b"])
#     logits[torch.where(logits<0)]=0
    dual[torch.where(dual<0)]=0
#     sample["label"][torch.where(sample["label"]<=-1e4)]=0
    err1 = torch.linalg.norm(logits - dataset["label"].unsqueeze(1))
    err3 = torch.linalg.norm(dual - dataset["dual"].unsqueeze(1))
    err2 = torch.linalg.norm(logits - dataset["label"].unsqueeze(1), ord=np.inf)
    err4 = torch.linalg.norm(dual - dataset["dual"].unsqueeze(1), ord=np.inf)
    return logits, dual, err1, err2, err3, err4, time11,names
def generate_pagerank_mps_dataset(num_nodes, degree, damping_factor, output_filename, seed=None):
    #     np.random.seed(seed)
    #     random.seed(seed)
    set_seed(seed)
    model = model_builder.ModelBuilder()
    print('Generating Graph', pcs.memory_info().rss)
    graph = nx.barabasi_albert_graph(num_nodes, degree)
    print('Graph generated', pcs.memory_info().rss)
    adjacency_matrix = nx.adjacency_matrix(graph)
    print('Adj Mat generated', pcs.memory_info().rss)
    stochastic_matrix_prime = adjacency_matrix / adjacency_matrix.sum(axis=0)
    adjacency_matrix = None
    print('Matrix Normalized ', pcs.memory_info().rss)
    lp_coefficients = damping_factor * stochastic_matrix_prime
    print('Coefficient Finished', pcs.memory_info().rss)
    lp_coefficients = lp_coefficients.tocsr()
    print(type(lp_coefficients))
    stt = lp_coefficients.indptr
    idx = lp_coefficients.indices
    val = lp_coefficients.data
    var_maps = {}
    for i in range(num_nodes):
        if i % 100 == 0:
            print(f'adding var: --------x_{i}/{num_nodes}---------', pcs.memory_info().rss)
        var_maps[f'x_{i}'] = model.new_var(name=f'x_{i}', lb=0.0, ub=1e+20, is_integer=False)
    print('Finished adding Vars')
    rhs = -(1 - damping_factor) / num_nodes
    for i in range(num_nodes):
        if i % 100 == 0:
            print(f'adding cons PRC: --------{i}/{num_nodes}---------')
        st = stt[i]
        ed = stt[i + 1]
        terms = []
        coeffs = []
        flag = None
        for ji, j in enumerate(idx[st:ed]):
            terms.append(var_maps[f'x_{j}'])
            if j == i:
                flag = i
            coeffs.append(val[st + ji])
        if flag is None:
            terms.append(var_maps[f'x_{i}'])
            coeffs.append(-1.0)
        else:
            coeffs[flag] -= 1.0
        #check 
        maps=set()
        for v in terms:
            if v.name not in maps:
                maps.add(v.name)
            else:
                print('Duplicated')
#                 quit()
        model.add(model_builder.LinearExpr().weighted_sum(terms, coeffs) <= rhs, name=f'PRC_{i}')
        # tmoModel.addCons(tmp_expr <= -(1 - damping_factor) / num_nodes,name=f'PRC_{i}')
    exps = []
    print('Finished adding Cons  1')
    for i in range(num_nodes):
        exps.append(var_maps[f'x_{i}'])
    model.add(model_builder.LinearExpr().sum(exps) == 1.0, name=f'SING')
    print('Finished adding Cons  2')
    return model

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def get_prob(prt,realx_int=False,include_name=True):
    return pdlp.qp_from_mpmodel_proto(prt,realx_int,include_name)

def getFeat(model, solve=True):
    data = {}
    nc = model.num_constraints
    nv = model.num_variables
    var_names=[]
    con_names=[]
    b_coef1=model.get_linear_constraint_lower_bounds().values
    b_coef2=model.get_linear_constraint_upper_bounds().values
#     isnan1 = np.isnan(b1)
#     isnan2 = np.isnan(b2)
#     isinf1 = np.isinf(b1)
#     isinf2 = np.isinf(b2)
#     if True in isnan1 or True in isinf1 :
#         b_coef = b2
#     if True in isnan2 or True in isinf2:
#         b_coef = b1
    b_coef=[]
    for i in range(len(b_coef1)):
        if np.isinf(b_coef1[i]):
            b_coef.append(b_coef2[i])
        else:
            b_coef.append(b_coef1[i])
    solver = model_builder.ModelSolver("PDLP")
#     solver.SetPrimalTolerance(1e-8)

    solver.enable_output(True)

    vs = model.get_variables()
    for v in vs:
        var_names.append(v.name)

    v_map = {}
    v_feat = []

    inf = float('inf')
    c_obj = []
    for v in vs:
        vname = v.name
        vidx = v.index
        lb = v.lower_bound
        ub = v.upper_bound
        c = v.objective_coefficient
        if lb == -inf:
            lb = -1e+20
        if ub == inf:
            ub = 1e+20
        v_map[vname] = vidx
        v_feat.append([lb, ub, c])
        c_obj.append(c)
    # need to convert v_feat to either numpy or torch tensor

    c_map = {}
    c_feat = []

    # nvar X ncon
    e_indx = [[], []]
    e_val = []

    cs = model.get_linear_constraints()
    
    for c in cs:
        con_names.append(c.name)
    exprs = model.get_linear_constraint_expressions()
    for idx, c in enumerate(cs):
        act_idx = c.index
        cn = c.name
        c_map[cn] = idx
        lb = c.lower_bound
        ub = c.upper_bound
        tmp = [0, 0, 0, None]
        # leq eq geq lb ub
        if lb <= -1e+20 and ub < 1e+20:
            tmp[0] = 1.0
            tmp[3] = ub
            # tmp[4] = ub
        elif ub >= 1e+20 and lb > -1e+20:
            tmp[2] = 1.0
            tmp[3] = lb
            # tmp[4] = 1e+20
        elif lb == ub:
            tmp[1] = 1.0
            tmp[3] = lb
            # tmp[4] = lb
        elif ub != lb:
            print('ERROR::unhandled constraint')
            print(lb, "<=", cn, "<=", ub)
            quit()
        c_feat.append(tmp)
        expr = exprs[act_idx]
        tm = expr.variable_indices
        coef = expr.coefficients
        for ii in range(len(tm)):
            e_indx[0].append(tm[ii])
            e_indx[1].append(act_idx)
            e_val.append(coef[ii])
    print(f'processed file with {nv}vars, {nc}cons, {len(e_val)}nnz')

    # add "pos emb" for each
    # -- v
    eps = 0.5
    dim = 16
    for i in range(len(v_feat)):
        tmp = [0] * (dim * len(v_feat[i]))
        for k in range(len(v_feat[i])):
            for j in range(dim // 2):
                tmp[dim * k + 2 * j] = math.sin(eps * v_feat[i][k] / math.pow(10000, 2 * j / dim))
        v_feat[i] = tmp
#         print(tmp)
    # -- c
    eps = 0.5
    dim = 16
    for i in range(len(c_feat)):
        tmp = [0] * (dim * len(c_feat[i]))
        for k in range(len(c_feat[i])):
            for j in range(dim // 2):
                tmp[dim * k + 2 * j] = math.sin(eps * c_feat[i][k] / math.pow(10000, 2 * j / dim))
        c_feat[i] = tmp
    dual_map = []
    sol_map = []
    # solving:
    if solve:
        print('Start solving using PDLP')
        solve_status = solver.solve(model)
        if mbh.SolveStatus.OPTIMAL == solve_status:
            #             print("Solved to optimal")
            #             print("Optimal objective value: ", solver.objective_value)
            variables = [model.var_from_index(i) for i in range(model.num_variables)]
            for c in model._get_linear_constraints(): 
                dual_map.append(solver.dual_value(c))
            for variable in variables:
                # print("index/name/value:", variable.index, variable.name, solver.value(variable))
                sol_map.append(solver.value(variable))
        else:
            print(solve_status)
            quit()
        print('Finished solving')
    data["edge_index"] = torch.tensor(e_indx, dtype=torch.int32).t().long()
    data["edge_weight"] = torch.tensor(e_val, dtype=torch.float32)
    data["con_feat"] = torch.tensor(c_feat, dtype=torch.float32)
    data["var_feat"] = torch.tensor(v_feat, dtype=torch.float32)
    data["b"] = torch.tensor(np.array(b_coef).reshape(data["con_feat"].shape[0], 1), dtype=torch.float32)
#     print(np.array(coef))
    data["c"] = torch.tensor(np.array(c_obj).reshape(data["var_feat"].shape[0], 1), dtype=torch.float32)
    data["label"] = torch.tensor(sol_map, dtype=torch.float32)
    data["dual"] = torch.tensor(dual_map, dtype=torch.float32)
    data["names"]=[var_names,con_names]
    # return v_feat, c_feat, e_indx, e_val, sol_map,coef,b_coef
    return data


def set_from_model2(x,y,model):
    ptc=model.export_to_proto()
    mdl = get_prob(ptc)
    new_sol_map=[]
    new_dual_map=[]
#     for idx,vn in enumerate(mdl.variable_names):
#         if 'u' in vn:
#             new_sol_map.append(0.0)
#         else:
#             new_sol_map.append(x[idx])
    for idx,vn in enumerate(mdl.variable_names):
#         if 'u' in vn:
#             new_sol_map.append(0.0)
#             continue
        new_sol_map.append(x[vn])
#         new_sol_map.append(0.0)
#     print(y)
#     print(y[:100])
#     quit()
    for idx,cn in enumerate(mdl.constraint_names):
        new_dual_map.append(y[cn])
#         new_dual_map.append(0.0)

    params = solvers_pb2.PrimalDualHybridGradientParams()
#     for e in dir(solvers_pb2):
#         print(e)
#     quit()
    opt_criteria = params.termination_criteria.simple_optimality_criteria
    opt_criteria.eps_optimal_relative = 1.0e-4
#     opt_criteria.eps_optimal_absolute = 1.0e-6
#     params.l_inf_ruiz_iterations = 0
#     params.l2_norm_rescaling = False
    start = pdlp.PrimalAndDualSolution()
    start.primal_solution = new_sol_map
    start.dual_solution = new_dual_map
    result = pdlp.primal_dual_hybrid_gradient(
        mdl, params, initial_solution=start
    )
    # for e in dir(result):
    #     print(e)
    sol_info=result.solve_log.solution_stats.convergence_information
    print('Used Primal/Dual Hint:')
    print('    Time:',round(result.solve_log.solve_time_sec,8),'    #iteration: ',result.solve_log.iteration_count)
    print(result.solve_log.solution_stats)
    
    return round(result.solve_log.solve_time_sec,8),result.solve_log.iteration_count
    

def solve_get_sol(model):
#     solver.enable_output(True)
    solver = model_builder.ModelSolver("PDLP")

    print('Start solving using PDLP')
    solve_status = solver.solve(model)
    sol_map={}
    dual_map={}
    if mbh.SolveStatus.OPTIMAL == solve_status:
        #             print("Solved to optimal")
        #             print("Optimal objective value: ", solver.objective_value)
        variables = [model.var_from_index(i) for i in range(model.num_variables)]
        for c in model._get_linear_constraints(): 
            dual_map[c.name]=solver.dual_value(c)
        for variable in variables:
            # print("index/name/value:", variable.index, variable.name, solver.value(variable))
            sol_map[variable.name]=solver.value(variable)
    print('Finished solving')
    return sol_map,dual_map


def set_from_model(x,y,model):
    ptc=model.export_to_proto()
    mdl = get_prob(ptc)
    params = solvers_pb2.PrimalDualHybridGradientParams()
    opt_criteria = params.termination_criteria.simple_optimality_criteria
    opt_criteria.eps_optimal_relative = 1.0e-4
    start = pdlp.PrimalAndDualSolution()
    start.primal_solution = x
    start.dual_solution = y
    result = pdlp.primal_dual_hybrid_gradient(
        mdl, params, initial_solution=start
    )
    # for e in dir(result):
    #     print(e)
    sol_info=result.solve_log.solution_stats.convergence_information
    print('Used Primal/Dual Hint:')
    print('    Time:',round(result.solve_log.solve_time_sec,8),'    #iteration: ',result.solve_log.iteration_count)
    print(sol_info)
    
    return round(result.solve_log.solve_time_sec,8),result.solve_log.iteration_count
    
    
def set_model(model):
    ptc=model.export_to_proto()
    mdl = get_prob(ptc)
    params = solvers_pb2.PrimalDualHybridGradientParams()
    opt_criteria = params.termination_criteria.simple_optimality_criteria
    opt_criteria.eps_optimal_relative = 1.0e-4
    result = pdlp.primal_dual_hybrid_gradient(
        mdl, params
    )
    # print(result.solve_log)
    sol_info=result.solve_log.solution_stats.convergence_information
    print('Without Primal/Dual Hint:')
    print('     Time:',round(result.solve_log.solve_time_sec,8),'    #iteration: ',result.solve_log.iteration_count)
    print(result.solve_log.solution_stats)
    return round(result.solve_log.solve_time_sec,8),result.solve_log.iteration_count

    


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


stp=args.stepsize
pwd=args.primalweight

n_files = [10, args.ntest]
num_nodes = 1000000
degree = 3
damping_factor = 0.85
dataset = []    
fp=gzip.open('../data/ip/lp_1.pkl','rb')
pk=pickle.load(fp)
fp.close()
print('Pickle Loaded')
dataset.append([pk[0], None])
    
    
# n_Samples = 12345678
# n_Samples = 'ip1e5'
n_Samples = 'ip1e5_1000_new_1000'
n_Samples = args.ident
# n_Samples = 10000
# print(dataset)
m = dataset[0][0]["b"].shape[0]  # Each LP has 10 constraints
n = dataset[0][0]["c"].shape[0]  # Each LP has 50 variables
nnz = dataset[0][0]["edge_index"].shape[1]  # Each LP has 100 nonzeros in matrix A
nConsF = dataset[0][0]["con_feat"].shape[1]  # Each constraint has 2 dimension
nVarF = dataset[0][0]["var_feat"].shape[1]
# SET-UP MODEL
embSize = int(args.embSize)
model_path = '../models/'+ '_d' + str(n_Samples) + '_s' + str(embSize) + '.pkl'
if 'small' in n_Samples:
    model_path = '../models/'+ 'ip_small_' + str(n_Samples) + '_s' + str(embSize) + '.pkl'
else:
    model_path = '../models/'+ 'ip_' + str(n_Samples) + '_s' + str(embSize) + '.pkl'
# set_seed(0)
gpu_index = int(args.gpu)
# 设置要使用的GPU设备的索引
torch.cuda.set_device(gpu_index)
# 获取当前使用的GPU设备
current_device = torch.cuda.current_device()
print(f"Using GPU {current_device}")
# 启用GPU内存的动态分配
torch.cuda.empty_cache()  # 清空缓存，释放内存
print("torch.cuda.is_available():{}".format(torch.cuda.is_available()))

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

model = GCNPolicy(embSize, nConsF, nVarF, isGraphLevel=False)
model.load_state_dict(torch.load(model_path))
model.to(device)  # Move the model to GPU if available
# TEST MODEL
start_time = time.time()
# solution = process(model, dataset, device)
predict_time = 0
PDLP_time = 0
search_time = 0
a, b, c,d,err5453, err4354, time01,dsad = process(model, dataset[0][0], device)
time02 = time.time()

logging.info(f"第0个数据集时间是{time02 - time01}")

l2loss_list =[]
inf_loss_list =[]
dual_l2loss_list =[]
dual_inf_loss_list =[]
L2O_time_list = []
PDLP_time_list =[]
GNN_time_list = []
PDLP_iter_list =[]
GNN_iter_list = []
for i in range(n_files[1]):
    if True:
        knapsack = GenerateKnapsack_simp(random_seed=i+10000)
        modelz = BuildMipForKnapsack(knapsack)
        data = getFeat(modelz.clone(), True)
    time11 = time.time()
    solution, dual, err1, err2,err3, err4, time11,names = process(model, data, device)
    time12 = time.time()
    
    l2loss_list.append(err1.item())
    inf_loss_list.append(err2.item())
    dual_l2loss_list.append(err3.item())
    dual_inf_loss_list.append(err4.item())
    val_list = []
    sol_map = solution.squeeze()
    dual_map = dual.squeeze()
    x={}
    y={}
    for idx,val in enumerate(sol_map):
        x[names[0][idx]]=val.item()
    for idx,val in enumerate(dual_map):
        y[names[1][idx]]=val.item()
    time3,iter3 = set_model(modelz.clone())
    time2,iter2 = set_from_model2(x,y,modelz.clone())
    predict_time += time12 - time11
    PDLP_time += time3
    search_time += time2
    L2O_time_list.append(time2)
    PDLP_time_list.append(time3)
    GNN_time_list.append(time12 - time11)
    PDLP_iter_list.append(iter3)
    GNN_iter_list.append(iter2)
    print( time12 - time11)
    logging.info(
    f"Project execution time: predict:{time12 - time11} seconds, search:{time2} seconds, our_total:{time12 - time11+time2} seconds, PDLP:{time3} seconds")
    print( f"Project execution time: predict:{time12 - time11} seconds, search:{time2} seconds, our_total:{time12 - time11+time2} seconds, PDLP:{time3} seconds")
    modelz=None
logging.info("********************************************************")
logging.info(f"Project execution time: predict:{predict_time} seconds, search:{search_time} seconds, our_total:{predict_time + search_time} seconds, PDLP:{PDLP_time} seconds")
print(
    "********************************************************")
print(
    f"Project execution time: predict:{predict_time} seconds, search:{search_time} seconds, our_total:{predict_time + search_time} seconds, PDLP:{PDLP_time} seconds")

name = "IP"
if 'small' in n_Samples:
    name='ip_small'

log_path = '../records/{args.code}/'+f"test_time_{stp}_{pwd}"+ name+'.csv'


import pandas as pd

df1 = pd.DataFrame({'l2_loss':l2loss_list,"infloss": inf_loss_list,"dual_l2loss":dual_l2loss_list, "dual_inf_loss": dual_inf_loss_list, "GNN_time":GNN_time_list, "L2O_time": L2O_time_list, "PDLP_time":PDLP_time_list,"L2O_iter":GNN_iter_list,"PDLP_iter":PDLP_iter_list})
df1.to_csv(log_path,index=False)



