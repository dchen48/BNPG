import torch
import numpy as np
import torch.nn.functional as F
import copy
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

class Net(nn.Module):
    def __init__(self, n, policy = 'tabular', G_type = 'all_zeros', device = 'cpu'):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.policy = policy
        self.n = n
        self.device = device
        if G_type == 'all_ones':
            self.G = torch.triu(torch.ones(self.n, self.n), diagonal=1).int()
        elif G_type == 'all_zeros':
            self.G = torch.zeros(self.n, self.n).int()
        elif G_type == 'line':
            self.G = self.get_line(n).int()
        else:
            raise NotImplementedError("Not implemented")
        self.size_s = 2**n
        self.size_a = 2**n
        self.parents = {}
        if self.policy == 'tabular':
            self.m = nn.Parameter(torch.empty(2**n, 2).normal_(mean=0,std=1), requires_grad=True)
        elif self.policy == 'tabular_baysian':
            parameter_lists = []
            for i in range(self.n):
                parents = set()
                for j in range(self.n):
                    if self.G[:,i][j]==1:
                        parents.add(j)
                self.parents[i] = parents
                num_parents = len(parents)
                parameter_lists.append(nn.Parameter(torch.empty(2**n, 2**num_parents, 2).normal_(mean=0,std=1), requires_grad=True))
            self.m = nn.ParameterList(parameter_lists)
        else:
            raise NotImplementedError("Not implemented")
    
    def helper(self, i):
        action_probs_i = self.softmax(self.m[i].view(-1,2)) #s * num_parents * 2
        action_probs_i = action_probs_i.view(self.size_s, 2**(len(self.parents[i])), 2)
        shape = [self.size_s]
        repeat_amounts = [1]
        for j in range(self.n):
            if j in self.parents[i]:
                shape.append(2)
                repeat_amounts.append(1)
            elif j !=i:
                shape.append(1)
                repeat_amounts.append(2)
            else:
                continue
        shape.append(2)
        repeat_amounts.append(1)
        action_probs_i = action_probs_i.view(shape)
        action_probs_i = action_probs_i.repeat(repeat_amounts)
        action_probs_i = action_probs_i.view(self.size_s, 2**i, 2**(self.n-i-1), 2)
        action_probs_i = action_probs_i.permute((0,1,3,2))
        action_probs_i = action_probs_i.reshape(self.size_s, self.size_a)
        return action_probs_i
        
    

    def get_line(self, n):
        line = torch.zeros((n,n))
        for i in range(n-2):
            line[i][i+1] = 1
        return line.int()

    def init(self):
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias)
        nn.init.normal_(self.fc3.weight)
        nn.init.normal_(self.fc3.bias)

    def softmax(self, x):
        z = x - torch.max(x,dim=1).values.view(len(x),1).repeat(1,len(x[0]))
        numerator = torch.exp(z)
        denominator = torch.sum(numerator,dim=1).view(len(x),1).repeat(1,len(x[0]))
        softmax = numerator/denominator
        return softmax

    def forward(self, x):
        if self.policy == 'tabular':
            return self.m
        elif self.policy == 'tabular_baysian':
            action_probs = torch.ones(self.size_s, self.size_a).to(self.device)
            for i in range(self.n):
                action_probs_i = self.softmax(self.m[i].view(-1,2)) #s * num_parents * 2
                action_probs_i = action_probs_i.view(self.size_s, 2**(len(self.parents[i])), 2)
                shape = [self.size_s]
                repeat_amounts = [1]
                for j in range(self.n):
                    if j in self.parents[i]:
                        shape.append(2)
                        repeat_amounts.append(1)
                    elif j !=i:
                        shape.append(1)
                        repeat_amounts.append(2)
                    else:
                        continue
                shape.append(2)
                repeat_amounts.append(1)
                action_probs_i = action_probs_i.view(shape)
                action_probs_i = action_probs_i.repeat(repeat_amounts)
                action_probs_i = action_probs_i.view(self.size_s, 2**i, 2**(self.n-i-1), 2)
                action_probs_i = action_probs_i.permute((0,1,3,2))
                action_probs_i = action_probs_i.reshape(self.size_s, self.size_a)
                action_probs = action_probs * action_probs_i
            return action_probs
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

class CoordinationGame():
    def __init__(self, n, eplison,gamma,mu,eta,policy = 'tabular', k = 1, lbda = 10, optimizer_type = 'SGD', device = 'cpu', G_type = 'all_ones'):
        self.n = n
        self.eta = eta
        if policy == 'tabular':
            self.clip_amount = 1e3
        else:
            self.clip_amount = 1e3
        self.device = device
        self.policy = policy
        self.lbda = lbda
        self.optimizer_type = optimizer_type
        self.G_type = G_type

        if self.policy == 'tabular_baysian':
            self.baysian_policy = Net(n, self.policy, G_type = self.G_type, device = self.device).to(self.device)
        else:
            raise NotImplementedError("Not implemented")
        self.parameters = []

        if self.policy == 'tabular_baysian':
            self.parameters = list(self.baysian_policy.parameters())
        else:
            raise NotImplementedError("Not implemented")

        self.optimizer = self.get_optimizer(self.optimizer_type, self.parameters)
        self.k = k
                
        self.eplison = eplison
        self.gamma=gamma
        self.mu = mu.to(self.device)
        self.prob=[[1-eplison,eplison],[eplison,1-eplison]] #p(s|a)
        self.trans_prob={(0,0):1-eplison,(0,1):eplison,(1,0):eplison,(1,1):1-eplison} #p(s|a)
        self.states_probs = None
        self.best_value = None
        self.size_s = 2**n
        self.size_a = 2**n
        self.l={2:1,3:1,5:2,10:3}
        self.reward_table = self.get_reward(self.l[self.n]) #2**n

    def get_reward(self,l):
        reward = torch.zeros(self.size_a)
        for i in range(self.size_a):
            state = np.binary_repr(i, self.n)
            if abs(state.count('0') - state.count('1'))<=l:
                if state.count('0') < state.count('1'):
                    reward[i] = 1
                else:
                    reward[i] = 0
            elif state.count('0') > state.count('1'):
                reward[i] = 3
            else:
                reward[i] = 2
        return reward.to(self.device)
    
    def get_optimizer(self, optimizer_type, parameters):
        if optimizer_type == 'SGD':
            optimizer = optim.SGD(parameters, lr=self.eta)
        elif optimizer_type == 'Adam':
            optimizer = optim.Adam(parameters, lr=self.eta)
        else:
            raise NotImplementedError("Not implemented")
        return optimizer

    
    def softmax(self, x):
        z = x - torch.max(x,dim=1).values.view(len(x),1).repeat(1,len(x[0]))
        numerator = torch.exp(z)
        denominator = torch.sum(numerator,dim=1).view(len(x),1).repeat(1,len(x[0]))
        softmax = numerator/denominator
        return softmax
    
    def array_rearange(self, A, size):
        ret = []
        m,n = A.shape
        l = int(m/size[0])
        p = int(n/size[1])
        for i in range(1,l+1):
            for j in range(1,p+1):
                x = A[(i-1)*size[0]:i*size[0],(j-1)*size[1]:j*size[1]].tolist()
                ret+= A[(i-1)*size[0]:i*size[0],(j-1)*size[1]:j*size[1]].tolist()
        return torch.tensor(ret).to(self.device)

    
    def get_p_pi(self, actions_probs):
        actions_probs = torch.unsqueeze(actions_probs, dim=0).repeat(self.size_a,1,1)
        states_probs = torch.unsqueeze(self.states_probs, dim=-1).repeat(1,1,self.size_a)
        p_pi = states_probs * actions_probs
        p_pi = torch.unsqueeze(p_pi, dim=0).repeat(self.size_s,1,1,1)
        p_pi = p_pi.view(self.size_s * self.size_a, self.size_s * self.size_a)
        return p_pi
    
    def get_p(self, actions_probs, states_probs):
        p = actions_probs @ states_probs
        return p
    
    def get_Q(self, actions_probs):
        r = self.reward_table.view(-1,1).repeat(1,self.size_a)
        r = r.view(-1,1)
        p_pi = self.get_p_pi(actions_probs)
        Q = torch.linalg.solve(torch.eye(self.size_s * self.size_a).to(self.device)-self.gamma*p_pi,r)
        return Q.view(self.size_s,self.size_a)
    
    def get_d(self, actions_probs):
        p = self.get_p(actions_probs, self.states_probs)
        temp1 = torch.linalg.solve((torch.eye(2**self.n).to(self.device)-self.gamma*p).T,self.mu).T
        d = (1-self.gamma)*(temp1).view(-1)
        return d
    
    def get_V(self,Q, actions_probs):
        return (Q * actions_probs).sum(dim=1)
    
    def get_Q_i(self, i, Q, actions_probs): #s*2
        Q_i = 0
        Q_temp = torch.zeros(self.size_s, int(self.size_a/2),2).to(self.device)
        a_temp = torch.zeros(self.size_a, int(self.size_a/2),2).to(self.device)
        actions_except_i_probs = actions_probs.view(self.size_s, 2**i,2,2**(self.n-i-1))
        actions_except_i_probs = actions_except_i_probs.permute((0,2,1,3))
        actions_except_i_probs = torch.sum(actions_except_i_probs,dim=1)
        actions_except_i_probs = actions_except_i_probs.view(self.size_s, 2**(self.n-1))
        for s in range(self.size_s):
            for a_m_i in range(int(self.size_a/2)):
                for a_i in range(2):
                    a_minus_i = np.binary_repr(a_m_i, self.n-1)
                    a = a_minus_i[0:i] + str(a_i) + a_minus_i[i:]
                    Q[s,int(a,2)]
                    Q_temp[s,a_m_i,a_i] = Q[s,int(a,2)]
                    a_temp[s,a_m_i,a_i] = actions_except_i_probs[s,a_m_i]
        Q_i = Q_temp * a_temp
        Q_i = Q_i.sum(dim=1)
        return Q_i, actions_except_i_probs
                                 
    def get_actions_probs(self, theta): #(s*(i,j,k...)*joint action)
        actions_probs = []
        probs_per_states = self.softmax(theta.view(-1,2)).view(self.size_s,-1,2)
        for probs in probs_per_states:
            probs = [prob.view(2,1) for prob in probs]
            actions_probs.append(self.get_joint_probs(probs).view(-1))
        actions_probs = torch.stack(actions_probs)
        return actions_probs, probs_per_states
    
    def get_states_probs(self): #(i,j,k...)*joint action * joint s
        probs = torch.stack([torch.tensor([1-self.eplison,self.eplison,self.eplison,1-self.eplison]).view(4,1) for i in range(self.n)]).to(self.device)
        states_probs = self.get_joint_probs(probs,rearrange=True).view(self.size_a,self.size_s)
        return states_probs.to(self.device)
    
    def get_states_probs_i(self): #(i,j,k...)*joint action * joint s
        states_probs_i = {}
        for i in range(self.n):
            probs = torch.stack([torch.tensor([1-self.eplison,self.eplison,self.eplison,1-self.eplison]).view(4,1) for k in range(self.n) if k!=i]).to(self.device)
            states_probs = self.get_joint_probs(probs,rearrange=True).view(2**(self.n-1), 2**(self.n-1))
            states_probs_i[i] = states_probs.to(self.device)
        return states_probs_i

    def value_iteration(self,P,r,eps = 1e-4):
        Q = torch.zeros(len(r)).to(self.device)
        next_Q = None
        V_Q = 0
        prev_V_Q = 0
        while True:
            prev_V_Q = V_Q
            V_Q = torch.max(Q.view(self.size_s,-1), dim=1).values
            V_Q = V_Q.view(-1,1)
            next_Q = r + self.gamma * P @ V_Q
            if torch.norm(next_Q - Q) <= eps:
                break
            Q = next_Q
        Q = Q.view(self.size_s, -1)
        x = torch.max(Q,dim=1)
        V_max = x.values
        return V_max
    
    def value_iteration_global(self,P,r,eps = 1e-4):
        Q = torch.zeros(len(r)).to(self.device)
        next_Q = None
        V_Q = 0
        prev_V_Q = 0
        while True:
            prev_V_Q = V_Q
            V_Q = torch.max(Q.view(self.size_s,-1), dim=1).values
            V_Q = V_Q.view(-1,1)
            PV_Q= P @ V_Q
            PV_Q=PV_Q.view(1,-1,1).repeat(self.size_s,1,1)
            PV_Q = PV_Q.view(-1,1)
            next_Q = r + self.gamma * PV_Q
            if torch.norm(next_Q - Q) <= eps:
                break
            Q = next_Q
        Q = Q.view(self.size_s, -1)
        x = torch.max(Q,dim=1)
        V_max = x.values
        return V_max
            
    def get_NE_Gap_i(self, i, actions_probs): #(i,j,k...)*joint action * joint s
        
        size = 2**(self.n-1)              #s*a
        #for P(s,a_i)
        action_probs_i = self.baysian_policy.helper(i)+1e-20

        actions_probs = actions_probs/action_probs_i
        actions_probs = actions_probs.view(self.size_s, 2**i,2,2**(self.n-i-1))
        actions_probs = actions_probs.permute((0,2,1,3))
        P = actions_probs.reshape(self.size_s, 2, -1)

        actions_probs = actions_probs.reshape(self.size_s, 2, -1,1)
        actions_probs = actions_probs.repeat(1,1,1,size)
        states_probs_i = self.states_probs_i[i]
        states_probs_i = states_probs_i.view(1,1,size,size).repeat(self.size_s,2,1,1)
        P = (states_probs_i * actions_probs).sum(dim=2)
        P = P.view(self.size_s,2,2**i,1,2**(self.n-i-1))
        P = P.repeat(1,1,1,2,1)

        p_ai = torch.tensor([1-self.eplison,self.eplison,self.eplison,1-self.eplison]).view(1,2,1,2,1)
        p_ai = p_ai.repeat(self.size_s, 1, 2**i, 1,2**(self.n-i-1)).to(self.device)
        P = P*p_ai
        P = P.view(self.size_s,2,self.size_s)
        P = P.view(-1,self.size_s)
        
        r = self.reward_table.view(self.size_s,1).repeat(1,2).view(-1,1) #r(s,a_i) ???
        
        V_max = self.value_iteration(P,r)
        NE_Gap_i = V_max.T@self.mu
        return NE_Gap_i

    def get_joint_probs(self, probs, rearrange = False):
        joint_probs = probs[0]
        for i, prob in enumerate(probs[1:]):
            joint_probs = joint_probs @ prob.T
            if rearrange:
                joint_probs = self.array_rearange(joint_probs, (2**(i+1),2))
            joint_probs = joint_probs.view(-1,1)
        return joint_probs

    def get_log_barrier_loss(self, probs_per_states):
        log_barrier_loss = self.lbda * torch.log(1e-10+probs_per_states)
        log_barrier_loss = log_barrier_loss.view(-1).sum()
        return log_barrier_loss/self.size_s/2
  
    def update_policy(self): #s*a
        if self.states_probs == None:
            self.states_probs = self.get_states_probs()
            self.states_probs_i = self.get_states_probs_i()
            
        if self.best_value == None:
            P = self.states_probs.view(self.size_a, self.size_s)
            r = self.reward_table.view(-1,1).repeat(1,self.size_a)
            r = r.view(-1,1)
            self.best_value = (self.value_iteration_global(P,r)).T @ self.mu
        
        self.optimizer.zero_grad()
        
        actions_probs = self.baysian_policy([None])
        Q = self.get_Q(actions_probs)
        V = self.get_V(Q, actions_probs)
        
        J = V.T@self.mu
        
        d = self.get_d(actions_probs)
        
        NE_Gap = -float('inf')
        for i in range(self.n):
            NE_Gap = max(NE_Gap, self.get_NE_Gap_i(i, actions_probs.clone()))
        

        d = d.view(self.size_s, 1).repeat(1, self.size_a).detach()
        Q = Q.detach()
        
        loss = -1/(1-self.gamma)*d*Q*actions_probs
        loss = torch.sum(loss)

        loss.backward()
        unclipped_norm = clip_grad_norm_(self.parameters, self.clip_amount)
        self.optimizer.step()
       
        NE_Gap -= J
        return J.detach(), (J/self.best_value).detach(), NE_Gap.detach()
