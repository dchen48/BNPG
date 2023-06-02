import torch
import os
import json
import numpy as np
import cg

seeds = [i for i in range(1,50)]
ns = [2,3,5]
policy = 'tabular_baysian'
optimizer_type = 'SGD'
device ='cuda'

num_iterations = 2000
iterations = [i for i in range(num_iterations)]

if not os.path.exists('./results'):
    os.makedirs('./results')
for n in ns:
    poa_ones = []
    ne_gap_ones = []
    poa_zeros = []
    ne_gap_zeros = []
    poa_line = []
    ne_gap_line = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        
        mu = torch.tensor([1/n for i in range(2**n)]).view(-1,1)
        CoordinationGame_PG_ones = cg.CoordinationGame(n,0.1,0.95,mu,0.01,policy=policy, device=device, optimizer_type=optimizer_type, G_type = 'all_ones')
        CoordinationGame_PG_zeros = cg.CoordinationGame(n,0.1,0.95,mu,0.01,policy=policy, device=device, optimizer_type=optimizer_type, G_type = 'all_zeros')
        CoordinationGame_PG_line = cg.CoordinationGame(n,0.1,0.95,mu,0.01,policy=policy, device=device, optimizer_type=optimizer_type, G_type = 'line')

        poa_ones_seed = []
        poa_zeros_seed = []
        poa_line_seed = []

        ne_gap_ones_seed = []
        ne_gap_zeros_seed = []
        ne_gap_line_seed = []

        for i in range(num_iterations):
            V_i_ones_seed, POA_i_ones_seed, ne_gap_i_ones_seed = CoordinationGame_PG_ones.update_policy()
            poa_ones_seed.append(POA_i_ones_seed.item())
            ne_gap_ones_seed.append(ne_gap_i_ones_seed.item())


            V_i_zeros_seed, POA_i_zeros_seed, ne_gap_i_zeros_seed = CoordinationGame_PG_zeros.update_policy()
            poa_zeros_seed.append(POA_i_zeros_seed.item())
            ne_gap_zeros_seed.append(ne_gap_i_zeros_seed.item())

            V_i_line_seed, POA_i_line_seed, ne_gap_i_line_seed = CoordinationGame_PG_line.update_policy()
            poa_line_seed.append(POA_i_line_seed.item())
            ne_gap_line_seed.append(ne_gap_i_line_seed.item())
        
        poa_ones.append(poa_ones_seed)
        poa_zeros.append(poa_zeros_seed)
        poa_line.append(poa_line_seed)

        ne_gap_ones.append(ne_gap_ones_seed)
        ne_gap_zeros.append(ne_gap_zeros_seed)
        ne_gap_line.append(ne_gap_line_seed)

    if not os.path.exists('./results/'+str(n)):
        os.makedirs('./results/'+str(n))
    with open('./results/'+str(n)+'/poa_ones.json', 'w') as fp:
        json.dump(poa_ones, fp)
    with open('./results/'+str(n)+'/poa_zeros.json', 'w') as fp:
        json.dump(poa_zeros, fp)
    with open('./results/'+str(n)+'/poa_line.json', 'w') as fp:
        json.dump(poa_line, fp)
    
    with open('./results/'+str(n)+'/ne_gap_ones.json', 'w') as fp:
        json.dump(ne_gap_ones, fp)
    with open('./results/'+str(n)+'/ne_gap_zeros.json', 'w') as fp:
        json.dump(ne_gap_zeros, fp)
    with open('./results/'+str(n)+'/ne_gap_line.json', 'w') as fp:
        json.dump(ne_gap_line, fp)
