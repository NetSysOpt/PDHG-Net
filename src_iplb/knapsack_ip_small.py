import logging
import random
# import params_pb2
import gzip

from google.protobuf import text_format
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver import pywraplp
from ortools.linear_solver.python import model_builder
from ortools.linear_solver.python import model_builder_helper as mbh

import time

multiplier = 5
bdim= 100
ip_param = [{"i_min": 100 * multiplier, "i_max": 100 * multiplier, "c_min": 1, "c_max": 1, "d_min": [0.5]*bdim, "d_max": [0.9]*bdim},
            {"i_min": 5 * multiplier, "i_max": 5 * multiplier, "c_min": 1, "c_max": 1, "d_min": [0.6]*bdim,"d_max": [1.0]*bdim}]
f_min = [2.0]*bdim
f_max = [2.0]*bdim
b_min = 10 * multiplier
b_max = 10 * multiplier



# multiplier = 10
# bdim= 400
#1, 3250 variables, 6105 constraints, and 427050 constraint matrix nonzeros
#4, 25000 variables, 24420 constraints, and 6760800 constraint 
#6, 50000 variables, 36630 constraints, and 15193800 constraint matrix nonzeros
#1,2,8,25
# multiplier = 15
# bdim= 200
# ip_param = [{"i_min": 100 * multiplier, "i_max": 100 * multiplier, "c_min": 1, "c_max": 1, "d_min": [0.5]*bdim, "d_max": [0.9]*bdim},
#             {"i_min": 5 * multiplier, "i_max": 5 * multiplier, "c_min": 1, "c_max": 1, "d_min": [0.6]*bdim,"d_max": [1.0]*bdim}]
# f_min = [2.0]*bdim
# f_max = [2.0]*bdim
# # f_min=[2.0, 2.0, 2.0]
# # f_max=[2.0, 2.0, 2.0]
# b_min = 10 * multiplier
# b_max = 10 * multiplier


def BuildMipForKnapsack(knapsack) -> linear_solver_pb2.MPModelProto:
    print('Building...')
    model = model_builder.ModelBuilder()

    num_items = len(knapsack.item)
    num_bins = len(knapsack.bin)
    num_resources = knapsack.r

    # place_vars[i][b] contains the index of a variable denoting whether to place
    # a copy of item i in bin b.
    place_vars = [[None] * num_bins for _ in range(num_items)]
    for i in range(num_items):
        for b in range(num_bins):
            vname = 'place_%d_%d' % (i, b)
            place_vars[i][b] = model.new_var(name=vname, lb=0.0, ub=1.0, is_integer=False)
    print('created VAR::::place_i_j...')

    # Place the appropriate number of copies for each item:
    # for i:
    #   sum_{b} place(i, b) = item(i).c
    for i in range(num_items):
        cname = 'copies_ct_%d' % i
        exps = []
        coeffs = []
        for b in range(num_bins):
            exps.append(place_vars[i][b])
            coeffs.append(1.0)
        model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) == knapsack.item[i].c, name=cname)
    print('created CON::::copies_ct_i...')

    # Ensure all items fit in their bins:
    # for b, r:
    #   sum_{i} item(i).d(r) * place(i, b) <= knapsack.bin(b).s(r)
    for b in range(num_bins):
        for r in range(num_resources):
            cname = 'supply_ct_%d_%d' % (b, r)
            exps = []
            coeffs = []
            for i in range(num_items):
                exps.append(place_vars[i][b])
                coeffs.append(knapsack.item[i].d[r])
            model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) <= knapsack.bin[b].s[r], name=cname)
    print('created CON::::supply_ct_i_j...')

    # deficit_vars[b][r] contains the index of a variable tracking the deficit
    # of the utilization of the resource r in bin b wrt the utilization expected
    # in a perfectly balanced placement.
    deficit_vars = [[None] * num_resources for _ in range(num_bins)]
    for b in range(num_bins):
        for r in range(num_resources):
            vname = 'deficit_%d_%d' % (b, r)
            deficit_vars[b][r] = model.new_var(name=vname, lb=0.0, ub=1.0, is_integer=False)
            deficit_vars[b][r].objective_coefficient = 1.0
    print('created VAR::::deficit_i_j...')

    # for b, r:
    #   1 - num_bins / t(r) * sum_{i} item(i).d(r) * place(i, b) <= deficit(b, r)
    #
    #   where t(r) is the total demand for resource r.
    for r in range(num_resources):
        t = sum(item.d[r] for item in knapsack.item)
        for b in range(num_bins):
            cname = 'deficit_ct_%d_%d' % (b, r)
            exps = []
            coeffs = []
            exps.append(deficit_vars[b][r])
            coeffs.append(1.0)
            for i in range(num_items):
                exps.append(place_vars[i][b])
                coeffs.append(knapsack.item[i].d[r] * num_bins / t)
            model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) >= 1.0, name=cname)
    print('created CON::::deficit_ct_i_j...')

    max_deficit_vars = [None] * num_resources
    for r in range(num_resources):
        vname = 'max_deficit_%d' % r
        max_deficit_vars[r] = model.new_var(name=vname, lb=0.0, ub=1.0, is_integer=False)
        max_deficit_vars[r].objective_coefficient = 10.0 * num_bins * num_resources

    print('created VAR::::max_deficit_r...')

    # for r, b:
    #   max_deficit(r) >= deficit(b, r)
    for b in range(num_bins):
        for r in range(num_resources):
            cname = 'max_deficit_ct_%d_%d' % (b, r)
            exps = []
            coeffs = []
            exps.append(max_deficit_vars[r])
            coeffs.append(1.0)
            exps.append(deficit_vars[b][r])
            coeffs.append(-1.0)
            model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) >= 0.0, name=cname)
    print('created CON::::max_deficit_ct_b_r...')

    print('Building finished.')
    return model


def BuildMipForKnapsack2(knapsack) -> linear_solver_pb2.MPModelProto:
    print('Building...')
    model_proto = linear_solver_pb2.MPModelProto()

    num_items = len(knapsack.item)
    num_bins = len(knapsack.bin)
    num_resources = knapsack.r

    # place_vars[i][b] contains the index of a variable denoting whether to place
    # a copy of item i in bin b.
    place_vars = [[None] * num_bins for _ in range(num_items)]
    for i in range(num_items):
        for b in range(num_bins):
            place_vars[i][b] = len(model_proto.variable)
            var_proto = model_proto.variable.add()
            var_proto.name = 'place_%d_%d' % (i, b)
            var_proto.is_integer = False
            var_proto.lower_bound = 0.0
            var_proto.upper_bound = 1.0
    print('created VAR::::place_i_j...')

    # Place the appropriate number of copies for each item:
    # for i:
    #   sum_{b} place(i, b) = item(i).c
    for i in range(num_items):
        ct_proto = model_proto.constraint.add()
        ct_proto.name = 'copies_ct_%d' % i
        ct_proto.lower_bound = knapsack.item[i].c
        ct_proto.upper_bound = knapsack.item[i].c
        for b in range(num_bins):
            ct_proto.var_index.append(place_vars[i][b])
            ct_proto.coefficient.append(1.0)
    print('created VAR::::copies_ct_i...')

    # Ensure all items fit in their bins:
    # for b, r:
    #   sum_{i} item(i).d(r) * place(i, b) <= knapsack.bin(b).s(r)
    for b in range(num_bins):
        for r in range(num_resources):
            ct_proto = model_proto.constraint.add()
            ct_proto.name = 'supply_ct_%d_%d' % (b, r)
            ct_proto.upper_bound = knapsack.bin[b].s[r]
            for i in range(num_items):
                ct_proto.var_index.append(place_vars[i][b])
                ct_proto.coefficient.append(knapsack.item[i].d[r])
    print('created CON::::supply_ct_i_j...')

    # deficit_vars[b][r] contains the index of a variable tracking the deficit
    # of the utilization of the resource r in bin b wrt the utilization expected
    # in a perfectly balanced placement.
    deficit_vars = [[None] * num_resources for _ in range(num_bins)]
    for b in range(num_bins):
        for r in range(num_resources):
            deficit_vars[b][r] = len(model_proto.variable)
            var_proto = model_proto.variable.add()
            var_proto.name = 'deficit_%d_%d' % (b, r)
            var_proto.is_integer = False
            var_proto.lower_bound = 0.0
            var_proto.upper_bound = 1.0
            var_proto.objective_coefficient = 1.0
    print('created CON::::deficit_i_j...')

    for r in range(num_resources):
        t = sum(item.d[r] for item in knapsack.item)
        for b in range(num_bins):
            ct_proto = model_proto.constraint.add()
            ct_proto.name = 'deficit_ct_%d_%d' % (b, r)
            ct_proto.lower_bound = 1.0
            ct_proto.var_index.append(deficit_vars[b][r])
            ct_proto.coefficient.append(1.0)
            for i in range(num_items):
                ct_proto.var_index.append(place_vars[i][b])
                ct_proto.coefficient.append(knapsack.item[i].d[r] * num_bins / t)
    print('created CON::::deficit_ct_i_j...')

    max_deficit_vars = [None] * num_resources
    for r in range(num_resources):
        max_deficit_vars[r] = len(model_proto.variable)
        var_proto = model_proto.variable.add()
        var_proto.name = 'max_deficit_%d' % r
        var_proto.is_integer = False
        var_proto.lower_bound = 0.0
        var_proto.upper_bound = 1.0
        var_proto.objective_coefficient = 10.0 * num_bins * num_resources
    print('created CON::::max_deficit_r...')

    for b in range(num_bins):
        for r in range(num_resources):
            ct_proto = model_proto.constraint.add()
            ct_proto.name = 'max_deficit_ct_%d_%d' % (b, r)
            ct_proto.lower_bound = 0.0
            ct_proto.var_index.append(max_deficit_vars[r])
            ct_proto.coefficient.append(1.0)
            ct_proto.var_index.append(deficit_vars[b][r])
            ct_proto.coefficient.append(-1.0)
    print('created CON::::max_deficit_ct_b_r...')

    print('Building finished.')
    return model_proto


class kbp:
    def __init__(self, r):
        self.r = r
        self.item = []
        self.bin = []


class itemZ:
    def __init__(self, c):
        self.c = c
        self.d = []


class binZ:
    def __init__(self):
        self.s = []


def GenerateKnapsack_simp(random_seed=0):
    print('Generating...')
    random.seed(random_seed)
    assert len(f_min) == len(f_max)
    knapsack = kbp(len(f_min))

    for item_param in ip_param:
        num_items = random.randint(item_param["i_min"], item_param["i_max"])
        for i in range(num_items):
            item = itemZ(random.randint(item_param["c_min"], item_param["c_max"]))
            item.d = [random.uniform(item_param["d_min"][r], item_param["d_max"][r])
                      for r in range(knapsack.r)]
            knapsack.item.append(item)

    b = random.randint(b_min, b_max)
    t = [sum(item.d[r] for item in knapsack.item) for r in range(knapsack.r)]
    f = [random.uniform(f_min[r], f_max[r])
         for r in range(knapsack.r)]

    for _ in range(b):
        tbin = binZ()
        tbin.s = [f[r] * t[r] / b for r in range(knapsack.r)]
        knapsack.bin.append(tbin)
    print('Generated.')
    return knapsack


def MPModelProtoToMPS(model_proto: linear_solver_pb2.MPModelProto):
    model_mps = pywraplp.ExportModelAsMpsFormat(model_proto)
    return model_mps


def solve_str(stdstr):
    model = model_builder.ModelBuilder()
    model.import_from_mps_string(stdstr)
    solver = model_builder.ModelSolver("PDLP")
    solver.set_solver_specific_parameters(
        'termination_criteria {simple_optimality_criteria {eps_optimal_relative:1e-4}}')
    print('Start solving using PDLP')
    solve_status = solver.solve(model)
    print(solve_status)
    if mbh.SolveStatus.OPTIMAL == solve_status:
        print("Solved to optimal")
        print("Optimal objective value: ", solver.objective_value)
    else:
        quit()


def solve(model):
    solver = model_builder.ModelSolver("PDLP")
    solver.set_solver_specific_parameters(
        'termination_criteria {simple_optimality_criteria {eps_optimal_relative:1e-8}}')
    print('Start solving using PDLP')
    solve_status = solver.solve(model)
    print(solve_status)
    if mbh.SolveStatus.OPTIMAL == solve_status:
        print("Solved to optimal")
        print("Optimal objective value: ", solver.objective_value)
    else:
        quit()


def gen_single(random_seed: int):
    logging.info('Building model , but not saving it')
    # t1=time.time()
    knapsack = GenerateKnapsack_simp(random_seed=random_seed)
    model = BuildMipForKnapsack(knapsack)
    return model


def generate(rs=0):
    return gen_single(rs)

def gen_t_ins(i):
    while True:
        knapsack = GenerateKnapsack_simp(random_seed=i)
        model = BuildMipForKnapsack(knapsack)
        if solve(model):
            break
    return model

# generate()
