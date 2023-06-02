from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn
from .models import model_factory
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.r_mappo.algorithm.probabilistic_dag_model.models import DeepSetOA

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, args, base, num_agents, action_space, inputs_dim, obs_info, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.base = base
        self.num_agents = num_agents
        self.args = args
        self.hidden_size = self.args.hidden_size
        self.recurrency = self.args.use_naive_recurrent_policy or self.args.use_recurrent_policy
        self._recurrent_N = self.args.recurrent_N
        self._use_orthogonal = self.args.use_orthogonal
        if self.recurrency:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        if action_space.__class__.__name__ == "Discrete":
            self.obs_dim = obs_info[0]
            self.obs_ally_dim = obs_info[1][1]
            self.action_dim = action_space.n
            self.sa_dim = self.obs_dim + self.action_dim
            self.net = DeepSetOA(self.sa_dim, inputs_dim)
            self.action_out = Categorical(inputs_dim, self.action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(
                inputs_dim, discrete_dim, use_orthogonal, gain)])
    
    def obs_to_bayesian_obs(self, obs, parent_action, indicies):
        bz = len(obs)
        obs_alley = obs[:,0:self.obs_ally_dim*(self.num_agents-1)].clone()
        obs_alley = obs_alley.view(bz, self.num_agents-1, self.obs_ally_dim).clone()
        obs_others = obs[:,self.obs_ally_dim*(self.num_agents-1):].clone()
        parent_indicies = torch.tensor([i for i in range(self.num_agents-1)]).unsqueeze(0).repeat(bz,1).cuda()
        agent_indicies =  indicies.unsqueeze(1).repeat(1, self.num_agents-1)
        parent_indicies = parent_indicies + (parent_indicies>=agent_indicies).int()
        parent_indicies = parent_indicies.unsqueeze(-1).repeat(1,1,self.action_dim)
        parent_indicies = parent_indicies
        is_alive_masks = obs_alley[:,:,0].unsqueeze(-1).repeat(1,1,self.action_dim)
        obs_alley_actions = torch.gather(parent_action, 1, parent_indicies) * is_alive_masks
        visible_dependency = torch.sum(obs_alley_actions)
        obs_alley[:,:,-self.action_dim:] = obs_alley_actions
        obs_alley = obs_alley.view(bz, -1)
        bayesian_obs = torch.cat((obs_alley, obs_others), dim=-1)
        return bayesian_obs, visible_dependency

    def forward(self, obs, x, G_s, G_s_list, rnn_states, rnn_masks, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        bz = len(obs)
        if available_actions != None:
            available_actions = available_actions.view(len(x), self.num_agents, -1)
        actions_outer = torch.zeros(bz, self.num_agents, 1, dtype=int).cuda()
        action_log_probs_outer = torch.zeros(bz, self.num_agents, 1).cuda()

        ordered_vertices_list = torch.tensor([G.topological_sorting() for G in G_s_list]).cuda()
        
        next_rnn_states = torch.zeros(rnn_states.shape).cuda()
        visible_dependencies = 0
        for step in range(self.num_agents):
            
            indicies = ordered_vertices_list[:,step]

            masks = torch.gather(G_s, 2, indicies.view(bz,1,1).repeat(1,self.num_agents,1))
            masks = masks.repeat(1,1, self.action_dim).cuda()
            parent_a = torch.eye(self.action_dim).cuda()[torch.arange(self.action_dim), actions_outer]
            parent_a = parent_a * masks
       
            current_o = obs[torch.arange(len(x)), indicies, :]

            current_bayesian_o, visible_dependency = self.obs_to_bayesian_obs(current_o, parent_a, indicies)
            visible_dependencies += visible_dependency
            actor_embd = self.base(current_bayesian_o)
            if self.recurrency:
                rnn_states_i = rnn_states[torch.arange(len(x)), indicies, :]
                rnn_masks_i = rnn_masks[torch.arange(len(x)), indicies, :]
                actor_features, next_rnn_states_i = self.rnn(actor_embd, rnn_states_i, rnn_masks_i)
                next_rnn_states[torch.arange(len(x)), indicies, :] = next_rnn_states_i

            if available_actions != None:
                available_actions_step = available_actions[torch.arange(len(available_actions)), indicies, :]
            else:
                available_actions_step = None
            action_logit = self.action_out(actor_embd, available_actions_step)
            action = action_logit.mode() if deterministic else action_logit.sample()
            actions_outer[torch.arange(len(x)), indicies,:] = action
            action_log_prob = action_logit.log_probs(action)
            action_log_probs_outer[torch.arange(len(x)), indicies,:] = action_log_prob
        
        father_action_lst_outer = torch.eye(self.action_dim).cuda()[torch.arange(self.action_dim), actions_outer.view(-1, 1).long()]
        father_action_lst_outer = father_action_lst_outer.view(bz, 1, self.num_agents * self.action_dim).repeat(1, self.num_agents, 1)
        visible_open_ratios = 2*visible_dependencies/len(obs)/(self.num_agents*(self.num_agents-1))
        return actions_outer.view(-1,1), action_log_probs_outer.view(-1,1), next_rnn_states, father_action_lst_outer.view(-1, self.num_agents * self.action_dim), visible_open_ratios

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, agent_id_batch, obs, x, G_s, G_s_list, rnn_states, rnn_masks, action_i, father_action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b] 
            action_log_probs = [] 
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum()) 
                    else:
                        dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
                
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = sum(dist_entropy)/len(dist_entropy)
        
        else:
            bz = len(obs)
            agent_id_batch = torch.tensor(agent_id_batch).view(-1).cuda()
            
            father_action = father_action.view(bz, self.num_agents, self.action_dim)
            
            masks = torch.gather(G_s, 2, agent_id_batch.view(bz,1,1).repeat(1,self.num_agents,1))
            masks = masks.repeat(1,1, self.action_dim).cuda()

      
            parent_a = father_action * masks
      
            bayesian_o, _ = self.obs_to_bayesian_obs(obs, parent_a, agent_id_batch)

            actor_embd = self.base(bayesian_o)
            if self.recurrency:
                actor_embd, _ = self.rnn(actor_embd, rnn_states, rnn_masks)
            
            action_logit = self.action_out(actor_embd, available_actions)
            action_logit_entropy = action_logit.entropy()
            action_log_prob = action_logit.log_probs(action_i)

            if active_masks is not None:
                dist_entropy = (action_logit_entropy*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logit_entropy.mean()
        
        return action_log_prob, dist_entropy
