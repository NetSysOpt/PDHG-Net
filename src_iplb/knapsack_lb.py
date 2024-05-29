import logging
import random
import gzip

from absl import app
from absl import flags

from google.protobuf import text_format
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver import pywraplp
import math
from ortools.linear_solver.python import model_builder
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
import numpy as np
from ortools.pdlp.python import pdlp

import time

multiplier = 2
# 2万 node
# worker_parameter = [
#     {"i_min": 200 * multiplier, "i_max": 200 * multiplier, "capacity_min": 0.5, "capacity_max": 0.8, "cost": 1},
#     {"i_min": 200 * multiplier, "i_max": 200 * multiplier, "capacity_min": 0.4, "capacity_max": 0.8, "cost": 1}
# ]

# workload_parameter = [
#     {"i_min": 40 * multiplier, "i_max": 40 * multiplier, "load_min": 1.0, "load_max": 5.0, "allowed_worker_probability": [0.15, 0.3]},
#     {"i_min": 10 * multiplier, "i_max": 10 * multiplier, "load_min": 5.0, "load_max": 10.0, "allowed_worker_probability": [0.3, 0.15]}
# ]

# sum prob_i*n_worker*cap_min

#20万node
worker_parameter = [
    {"i_min": 500 * multiplier, "i_max": 500 * multiplier, "capacity_min": 0.5, "capacity_max": 0.8, "cost": 1},
    {"i_min": 500 * multiplier, "i_max": 500 * multiplier, "capacity_min": 0.4, "capacity_max": 0.8, "cost": 1}
]

workload_parameter = [
    {"i_min": 100 * multiplier, "i_max": 100 * multiplier, "load_min": 1.0, "load_max": 5.0, "allowed_worker_probability": [0.15, 0.3]},
    {"i_min": 10 * multiplier, "i_max": 10 * multiplier, "load_min": 5.0, "load_max": 10.0, "allowed_worker_probability": [0.3, 0.15]}
]

def BuildMipForLoadBalancing(problem):
    print('Building...')
    model = model_builder.ModelBuilder()

    num_workloads = len(problem.workload)
    num_workers = len(problem.worker)

    # reserved_capacity_vars[s][t] contains the index of a continuous variable
    # denoting the capacity reserved on worker t for handling workload s.
    reserved_capacity_vars = [[None] * num_workers for _ in range(num_workloads)]
    for s in range(num_workloads):
        for t in range(num_workers):
            vname = f'reserved_capacity_{s}_{t}'
            reserved_capacity_vars[s][t] = model.new_var(name=vname, lb=0.0, ub=0.0, is_integer=False)
        # capacity can be reserved only on allowed workers.
        for t in problem.workload[s].allowed_workers:
            reserved_capacity_vars[s][t].upper_bound = problem.worker[t].capacity
    print(f'Finished adding reserved_capacity_s_t')

    # worker_used_vars[t] contains the index of a binary variable denoting whether
    # worker t is used in the solution.
    worker_used_vars = [None] * num_workers
    for t in range(num_workers):
        vname = f'worker_used_{t}'
        worker_used_vars[t] = model.new_var(name=vname, lb=0.0, ub=1.0, is_integer=False)
        worker_used_vars[t].objective_coefficient = problem.worker[t].cost
    print(f'Finished adding worker_used_t')

    # Capacity can be reserved only if a node is used.
    # For s, t: reserved_capacity_vars[s][t] <= M * worker_used_vars[t]
    # where M = max(worker[t].capacity, workload[s].load)
    for s in range(num_workloads):
        for t in range(num_workers):
            cname = f'worker_used_ct_{s}_{t}'
            exps = []
            coeffs = []
            exps.append(reserved_capacity_vars[s][t])
            coeffs.append(-1)
            exps.append(worker_used_vars[t])
            M = max(problem.worker[t].capacity, problem.workload[s].load)
            coeffs.append(M)
            model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) >= 0.0, name=cname)
    print(f'Finished adding worker_used_ct_s_t')

    # Cannot reserve more capacity than available.
    # For t: sum_s reserved_capacity_vars[s][t] <= worker[t].capacity
    for t in range(num_workers):
        cname = f'worker_capacity_ct_{t}'
        exps = []
        coeffs = []
        for s in range(num_workloads):
            exps.append(reserved_capacity_vars[s][t])
            coeffs.append(1)
        model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) <= problem.worker[t].capacity, name=cname)
    print(f'Finished adding worker_capacity_ct_t')

    # There must be sufficient capacity for each workload in the scenario where
    # any one of the allowed workers is unavailable.
    # For s, t: sum_{t' != t} reserved_capacity_vars[s][t'] >= workload[s].load
    for s in range(num_workloads):
        allowed_workers = sorted(set(problem.workload[s].allowed_workers))
        assert len(allowed_workers) > 1
        for unavailable_t in allowed_workers:
            cname = f'workload_ct_{s}_failure_{unavailable_t}'
            exps = []
            coeffs = []
            for t in allowed_workers:
                if t == unavailable_t:
                    continue
                exps.append(reserved_capacity_vars[s][t])
                coeffs.append(1)
            model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) >= problem.workload[s].load, name=cname)

    print(f'Finished adding workload_ct_s_failure_unavailable_t')

    return model


class wk:
    def __init__(self, cap, cost):
        self.capacity = cap
        self.cost = cost


class wl:
    def __init__(self, load):
        self.load = load
        self.allowed_workers = []


class lbP:
    def __init__(self):
        self.worker = []
        self.workload = []


def GenerateLoadBalancingProblem(random_seed=0):
    random.seed(random_seed)
    problem = lbP()

    # The index of the WorkerParameters that generated this worker.
    worker_group = []

    for (group, worker_param) in enumerate(worker_parameter):
        num_workers = random.randint(worker_param["i_min"], worker_param["i_max"])
        for _ in range(num_workers):
            worker = wk(random.uniform(worker_param["capacity_min"], worker_param["capacity_max"]), worker_param["cost"])
            worker_group.append(group)
            problem.worker.append(worker)

    for workload_param in workload_parameter:
        num_workloads = random.randint(workload_param["i_min"], workload_param["i_max"])
        assert len(workload_param["allowed_worker_probability"]) == len(worker_parameter)
        for _ in range(num_workloads):
            workload = wl(random.uniform(workload_param["load_min"], workload_param["load_max"]))
            for worker_index in range(len(problem.worker)):
                if random.random() < workload_param["allowed_worker_probability"][worker_group[worker_index]]:
                    workload.allowed_workers.append(worker_index)

            problem.workload.append(workload)

    return problem

def MPModelProtoToMPS(model_proto: linear_solver_pb2.MPModelProto):
    nnz = sum(len(c.var_index) for c in model_proto.constraint)
    logging.info('# vars = %d, # cons = %d, # nz = %d', len(model_proto.variable),
                 len(model_proto.constraint), nnz)
    model_mps = pywraplp.ExportModelAsMpsFormat(model_proto)
    return model_mps


def BuildRandomizedModels(random_seed=0):
    print('Generating')
    problem = GenerateLoadBalancingProblem(random_seed)
    model = BuildMipForLoadBalancing(problem)
    nc = model.num_constraints
    nv = model.num_variables
    print(f'Generated model with {nv}vars and {nc}cons')
    return model


def solve(model):
    solver = model_builder.ModelSolver("PDLP")
    solver.set_solver_specific_parameters(
            'termination_criteria {simple_optimality_criteria {eps_optimal_relative:1e-8}}')
    solver.enable_output(True)
    print('Start solving using PDLP')
    st_time = time.time()
    solve_status = solver.solve(model)
    timer = time.time() - st_time
    if mbh.SolveStatus.OPTIMAL == solve_status:
        print("Solved to optimal")
        print("Optimal objective value: ", solver.objective_value)
        print(f"              Wall Time: {timer}s")
        return True
    elif mbh.SolveStatus.INFEASIBLE == solve_status:
        print('ERROR::Solving terminated---------INFEASIBLE')
    elif mbh.SolveStatus.UNBOUNDED == solve_status:
        print('ERROR::Solving terminated---------UNBOUNDED')
        print(solve_status)
    elif mbh.SolveStatus.FEASIBLE == solve_status:
        print('ERROR::Solving terminated---------FEASIBLE')
        print(solve_status)
    elif mbh.SolveStatus.UNKNOWN_STATUS == solve_status:
        print('ERROR::Solving terminated---------UNKNOWN_STATUS')
        print(solve_status)
    elif mbh.SolveStatus.ABNORMAL == solve_status:
        print('ERROR::Solving terminated---------ABNORMAL')
        print(solve_status)
    elif mbh.SolveStatus.INCOMPATIBLE_OPTIONS == solve_status:
        print('ERROR::Solving terminated---------INCOMPATIBLE_OPTIONS')
        print(solve_status)
    elif mbh.SolveStatus.NOT_SOLVED == solve_status:
        print('ERROR::Solving terminated---------NOT_SOLVED')
        print(solve_status)
    else:
        print(solve_status)
    return False
        
def gen_t_ins(i):
    while True:
        model = BuildRandomizedModels(i)
        if solve(model):
            break
    return model

# model=BuildRandomizedModels()
# solve(model)
