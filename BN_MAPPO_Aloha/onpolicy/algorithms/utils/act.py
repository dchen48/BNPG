from .distributions import Bernoulli, Categorical, DiagGaussian
import torch as th
import torch.nn as nn
import numpy as np
from onpolicy.utils.util import *
import igraph as ig
from datetime import datetime, timedelta
import numpy as np


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, num_agents, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.num_agents = num_agents

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_dim = action_dim
            self.action_out_list = nn.ModuleList([Categorical(inputs_dim, action_dim, use_orthogonal, gain) for _ in range(self.num_agents)])
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
            self.action_outs = nn.ModuleList(
                [DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(
                    inputs_dim, discrete_dim, use_orthogonal, gain)])

    def forward(self, base, obs, x, G_s, available_actions=None, deterministic=False):
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
        actions_outer = torch.zeros(bz, self.num_agents, 1, dtype=int).cuda()
        action_log_probs_outer = torch.zeros(bz, self.num_agents, 1).cuda()
        
        for step in range(self.num_agents):
            indicies = {}
            masks = {}
            for id in range(self.num_agents): 
                indicies[id] = []
                masks[id] = []

            for i in range(len(G_s)):
                G = G_s[i]
                ordered_vertices = G.topological_sorting()
                j = ordered_vertices[step]
                indicies[j].append(i)
                masks[j].append(G[:,j])
            
            for agent_id in range(self.num_agents):
                if indicies[agent_id] == []:
                    continue
                masks_i = torch.tensor(masks[agent_id])
                father_action = actions_outer[indicies[agent_id],:,:].long()
                father_action = father_action.view(-1,1)
            
                father_action = torch.eye(self.action_dim).cuda()[torch.arange(self.action_dim), father_action]
                father_action = father_action.view(len(indicies[agent_id]), self.num_agents, -1)
                
                father_oa = torch.cat((obs[indicies[agent_id]], father_action), dim=2)
                masks_i = masks_i.unsqueeze(-1).repeat(1,1, father_oa.shape[-1]).cuda()
                father_oa = father_oa * masks_i

                father_oa = father_oa.view(len(indicies[agent_id]), -1)
                father_oa_embd = base(father_oa)
                actor_embd = torch.cat((x[indicies[agent_id],agent_id,:], father_oa_embd), dim=-1)
                
                action_logit = self.action_out_list[agent_id](actor_embd, available_actions)
                
                action = action_logit.mode() if deterministic else action_logit.sample()
                actions_outer[indicies[agent_id] , agent_id,:] = action
                action_log_prob = action_logit.log_probs(action)
                action_log_probs_outer[indicies[agent_id] , agent_id,:] = action_log_prob
             
        father_action_lst_outer = torch.eye(self.action_dim).cuda()[torch.arange(self.action_dim), actions_outer.view(-1, 1).long()]
        father_action_lst_outer = father_action_lst_outer.view(bz, 1, self.num_agents * self.action_dim).repeat(1, self.num_agents, 1)
        return actions_outer.view(-1,1), action_log_probs_outer.view(-1,1), father_action_lst_outer.view(-1, self.num_agents * self.action_dim)

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

    def evaluate_actions(self, base, x, agent_id_batch, father_action, action, available_actions=None, active_masks=None):
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
                        dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
                    else:
                        dist_entropy.append(
                            (action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
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
                    dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()

        else:
            indicies = {}
            for i in range(len(agent_id_batch)):
                idx = agent_id_batch[i][0]
                if idx not in indicies:
                    indicies[idx] = [i]
                else:
                    indicies[idx].append(i)
            action_log_probs = torch.zeros(len(agent_id_batch), 1).cuda()
            dist_entropy = 0  
            for i in range(self.num_agents):
                if i not in indicies:
                    continue
        
                father_oa_embd = base(father_action[indicies[i]])
                x_i = torch.cat((x[indicies[i]], father_oa_embd), dim=-1)
                action_logits_i = self.action_out_list[i](x_i)
                action[indicies[i]]
                action_log_probs_i = action_logits_i.log_probs(action[indicies[i]])
                action_log_probs[indicies[i]] = action_log_probs_i
                if active_masks is not None:
                    dist_entropy_i = (action_logits_i.entropy() * active_masks[[indicies[i]]].squeeze(-1)).sum() / active_masks[[indicies[i]]].sum()
                else:
                    dist_entropy_i = action_logits_i.entropy().mean()
                dist_entropy += dist_entropy_i

            dist_entropy = dist_entropy/self.num_agents
        return action_log_probs, dist_entropy

    def evaluate_baysian_actions(self, x, agent_id_batch, father_action, available_actions=None, active_masks=None):
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
                        dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
                    else:
                        dist_entropy.append(
                            (action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
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
                    dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()

        else:
            indicies = {}
            for i in range(len(agent_id_batch)):
                idx = agent_id_batch[i][0]
                if idx not in indicies:
                    indicies[idx] = [i]
                else:
                    indicies[idx].append(i)
            
            action_probs = torch.zeros(len(agent_id_batch), 2).cuda()
            dist_entropy = 0  

            for i in range(self.num_agents):
                if i not in indicies:
                    continue
                x_i = torch.cat((x[indicies[i]], father_action[indicies[i]]), dim=1)
                action_logits_i = self.action_out_list[i](x_i)
                action_probs_i = action_logits_i.probs
                action_probs[indicies[i]] = action_probs_i
                
        return action_probs
