from asyncio import base_tasks
import torch
from probabilistic_dag import ProbabilisticDAG

n_nodes=7
input_dim = 24
hidden_dim = 49
batch_size = 3000
permutation_net_type = 'set_transformer' #'deep_set'
model = ProbabilisticDAG(n_nodes, input_dim, hidden_dim, permutation_net_type = permutation_net_type)

max1 = -10
max2 = -10
for _ in range(5):
    x = []
    for _ in range(batch_size):
        idx = torch.randperm(n_nodes)
        A = torch.zeros(n_nodes,n_nodes)
        for i in range(n_nodes):
            A[i][idx[i]]=1
        x.append(A)
    x = torch.stack(x)

    obs = torch.rand(batch_size, n_nodes, input_dim)
    q = torch.rand(batch_size, n_nodes, n_nodes)
    P, U, dags, _, _ = model.sample(obs, fixed_edge_noise = 1.5, fixed_permutation_noise = q)

    P_p, U_p, dags_p, _, _ = model.sample(torch.bmm(x, obs), fixed_edge_noise = 100, fixed_permutation_noise = torch.bmm(x, q))
    
    
    a = torch.bmm(x, P)
    a = P_p - a
    a = torch.max(torch.abs(a))
    max1 = max(max1, a)

print(max1)



