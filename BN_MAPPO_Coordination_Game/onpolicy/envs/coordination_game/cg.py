from re import I
from typing import SupportsAbs
import torch
import numpy as np
import torch.nn.functional as F
import copy
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch import distributions
from gym import spaces
from .multi_discrete import MultiDiscrete
import copy

class CoordinationGame():
    def __init__(self, n, eplison, mu, gamma = 0.95, device = 'cpu'):
        self.n = n
        self.eplison = eplison
        self.gamma=gamma
        self.device = device
        self.mu = torch.tensor(mu).to(self.device)
        self.prob=[[1-eplison,eplison],[eplison,1-eplison]] #p(s|a)
        self.trans_prob={(0,0):1-eplison,(0,1):eplison,(1,0):eplison,(1,1):1-eplison} #p(s|a)
        self.states_probs = None
        self.best_value = None
        self.size_s = 2**n
        self.size_a = 2**n
        self.l={2:1,3:1,5:2,10:3}
        self.reward_table = self.get_reward(self.l[self.n]) #2**n
        
        self.world_length = 25
        self.current_step = 0
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.n,), dtype=np.float32) for _ in range(self.n)]
        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.n,), dtype=np.float32) for _ in range(self.n)]
        self.action_space = [spaces.Discrete(2) for _ in range(self.n)]


    def get_obs(self):
        obs = []
        for i in range(self.size_s):
            obs_i = np.binary_repr(i, self.n)
            obs_i = torch.tensor([float(x) for x in obs_i])
            obs.append(obs_i)
        obs = torch.stack(obs)
        return obs

    def get_parents(self, G):
        parents = {}
        for i in range(len(G)):
            if i not in parents:
                parents[i] = set()
            for j in range(len(G)):
                if G[j][i] == 1:
                    parents[i].add(j)
        return parents
        

    def get_father_action(self, agent_id, Gs):
        father_action = []
        Gs = Gs.int()
        for G in Gs:
            parents = self.get_parents(G)
            num_parents = int(torch.sum(G[:,agent_id]))
            father_action_i = torch.zeros(2**num_parents, self.n*2)
            for j in range(2**num_parents):
                parents_repr = np.binary_repr(j, num_parents)
                father_action_j = torch.zeros(self.n, 2)

                for k, f_c in zip(parents[agent_id],parents_repr):
                    f_c = int(f_c)
                    father_action_j[k] = torch.eye(2)[f_c]
                father_action_j = father_action_j.view(-1)
                father_action_i[j] = father_action_j
            shape = [1 for _ in range(self.n)]
            repeat = [2 for _ in range(self.n)]
            repeat[agent_id] = 1
            for k in parents[agent_id]:
                shape[k] = 2
                repeat[k] = 1
            shape.append(self.n * 2)
            repeat.append(1)
            father_action_i = father_action_i.view(shape)
            father_action_i = father_action_i.repeat(repeat)
            father_action_i = father_action_i.view(2**(self.n-1), self.n * 2)
            father_action.append(father_action_i)
              
        father_action = torch.stack(father_action)
        return father_action
    
    def get_baysian_probs(self, baysian_policy):
        action_probs = torch.ones(self.size_s, self.size_a).to(self.device)
        obs = self.get_obs()
        obs = obs.cuda()
        P, U, G_s, edge_noise, permutation_noise =  baysian_policy.dag_net.sample(obs, test = True)
        obs = obs.view(2**self.n, 1, self.n).repeat(1, 2**self.n, 1).view(-1, self.n).to(self.device)
        for i in range(self.n):
            agent_id_batch = np.array([[i] for _ in range(2**self.n * (2**(self.n-1)))])
            father_action = self.get_father_action(i, G_s)

            graph_obs_batch= obs
            last_actions_batch = None
            edge_noise_batch = None
            permutation_noise_batch = None
            rnn_states_actor = None
            graph_rnn_states_actor = None
            rnn_states_critic = None
            
            masks = 1
            graph_masks = 1      
            father_action = father_action.view(-1, self.n*2).to(self.device)
            
            action_probs_i = baysian_policy.evaluate_baysian_actions(obs, agent_id_batch, graph_obs_batch, father_action, last_actions_batch, edge_noise_batch, permutation_noise_batch, rnn_states_actor, graph_rnn_states_actor, rnn_states_critic, masks, graph_masks,
                         available_actions=None, graph_available_actions=None, active_masks=None)

            shape = [2 for _ in range(self.n)]
            shape[i] = 1
            shape = [2**self.n] + shape + [2]
            action_probs_i = action_probs_i.view(shape)
            permute_dim = [m for m in range(self.n+2)]
            permute_dim[i+1] = self.n+1
            permute_dim[self.n+1] = i+1
            action_probs_i = action_probs_i.permute(permute_dim)
            action_probs_i = action_probs_i.reshape(self.size_s, self.size_a)
            action_probs = action_probs*action_probs_i
        
        return action_probs
            



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
    
    def get_state_id(self, s): #bz*n
        bz, dim = s.shape
        state_id = []
        for a in s:
            a = a.tolist()
            str_a = ''.join([str(x) for x in a])
            int_a = int(str_a,2)
            state_id.append(int_a)
        return state_id
    
    def _get_done(self, agent):
        if self.current_step >= self.world_length:
            return True
        else:
            return False
        
            
    def step(self, action): #a 
        self.current_step += 1
        probs = torch.zeros((self.n, 2))
        done_n = []
        for i in range(len(probs)):
            probs[i][0] = self.trans_prob[(action[i][1].item(),0)]
            probs[i][1] = self.trans_prob[(action[i][1].item(),1)]
            done_n.append(self._get_done(i))

        next_state_dist = distributions.Categorical(probs) #action_dim = 2
        next_state = next_state_dist.sample()
        next_state = next_state.view(1, self.n)
        state_id = self.get_state_id(next_state)
        reward = self.reward_table[state_id]
        
        next_state = next_state.repeat(self.n, 1).float().tolist()
        reward = [list(reward) for _ in range(self.n)]
       
        return next_state, reward, done_n, None
    
    def reset(self):
        self.current_step = 0
        probs = torch.zeros(self.n, 2)
        for i in range(self.n):
            probs[i][0] = self.mu[i]
            probs[i][1] = 1 - self.mu[i]

        state_dist = distributions.Categorical(probs) 
        state = state_dist.sample()
        state = state.view(-1, self.n).repeat(self.n, 1).float().tolist()
        #return state.view(-1, self.n).float()
        return state

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

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
            torch.manual_seed(1)
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)

    
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
        #Q, _ = torch.solve(r, torch.eye(self.size_s * self.size_a).to(self.device)-self.gamma*p_pi)
        return Q.view(self.size_s,self.size_a)
    
    def get_d(self, actions_probs):
        p = self.get_p(actions_probs, self.states_probs)
        #temp = torch.linalg.solve((torch.eye(2**self.n).to(self.device)-self.gamma*p).T,self.mu).T
        #d = (1-self.gamma)*(temp).view(-1)
        temp1, _ = torch.solve(self.mu, (torch.eye(2**self.n).to(self.device)-self.gamma*p).T)
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
        states_except_self_probs = []
        size = 2**(self.n-1)
        for i in range(self.n):
            probs_except_i = [k for k in range(self.n) if k!=i]
        return states_probs.to(self.device)
    
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
            
    def get_NE_Gap_i(self, i, actions_except_i_probs): #(i,j,k...)*joint action * joint s
        
        size = 2**(self.n-1)
        
        #for P(s'|s,a_i)        
        actions_except_i_probs = actions_except_i_probs.view(self.size_s, size,1 ,1).repeat(1,1,2,self.size_s)
        states_probs = self.states_probs.view(2**i, 2, 2**(self.n-i-1),self.size_s)
        
        states_probs = states_probs.permute((0,2,1,3))
        states_probs = states_probs.reshape(1,size,2,self.size_s) #???
        states_probs = states_probs.repeat(self.size_s,1,1,1)
        P = actions_except_i_probs * states_probs
        P = torch.sum(P,dim=1)
        P = P.view(-1,self.size_s)
        #for P(s,a_i)
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
    
    def policy_clone(self):
        return [copy.deepcopy(policy) for policy in self.policies]
    
    def get_theta(self, policies):
        if not self.use_binary:
            states = torch.eye(self.size_s)
        else:
            states = []
            for i in range(self.size_s):
                state = np.binary_repr(i, self.n)
                binary_state = []
                for k in state:
                    k = int(k)
                    if k==0:
                        binary_state.append(0)
                        binary_state.append(1)
                    else:
                        binary_state.append(1)
                        binary_state.append(0)
                state = torch.tensor(binary_state)
                states.append(state)
            states = torch.stack(states).float()
        states = states.to(self.device)
        #states = torch.eye(2**self.n).to(self.device)
        theta = torch.stack([policy(states) for policy in policies])
        theta = theta.permute((1,0,2)).contiguous()
        return theta
        
    def get_log_barrier_loss(self, probs_per_states):
        log_barrier_loss = self.lbda * torch.log(1e-10+probs_per_states)
        log_barrier_loss = log_barrier_loss.view(-1).sum()
        return log_barrier_loss/self.size_s/2
        
    def get_poa(self, baysian_policy):
        
        if self.states_probs == None:
            self.states_probs = self.get_states_probs() 
        if self.best_value == None:
            P = self.states_probs.view(self.size_a, self.size_s)
            r = self.reward_table.view(-1,1).repeat(1,self.size_a)
            r = r.view(-1,1)
            self.best_value = (self.value_iteration_global(P,r)).T @ self.mu    
        
        actions_probs = self.get_baysian_probs(baysian_policy)
                
        Q = self.get_Q(actions_probs)
        V = self.get_V(Q, actions_probs)
        
        J = V.T@self.mu
        
        return (J/self.best_value).detach()
    
    def list_binary_2_int(self, l):
        return int("".join(str(i) for i in l),2)

    def get_baysian_V(self, baysian_policy, states): 
        
        if self.states_probs == None:
            self.states_probs = self.get_states_probs() 
        
        actions_probs = self.get_baysian_probs(baysian_policy)
        
        Q = self.get_Q(actions_probs)
        V = self.get_V(Q, actions_probs)
        
        states = states.astype(int)
        states = [self.list_binary_2_int(state.tolist()) for state in states]
      
        return V[states].view(-1,1).detach()
