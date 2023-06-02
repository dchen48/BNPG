import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.util import *
import igraph as ig
from onpolicy.algorithms.r_mappo.algorithm.probabilistic_dag_model.probabilistic_dag import *
from onpolicy.algorithms.utils.util import check
from gym import spaces

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.args = args
        self.game = None

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space

        if args.env_name == 'GRFootball':
            self.act_space = act_space
        else:
            self.act_space = act_space
        
        self.graph_type = self.args.graph_type
        self.threshold = self.args.threshold

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
      
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        
        self.num_actions = act_space.n
        
        if self.graph_type == 'dynamic':
            self.dag_net = ProbabilisticDAG(self.args.num_agents, self.obs_space.shape[0], args.hidden_size, temperature=1.0, hard=True, noise_factor=1.0, edge_net_type = args.edge_net_type, device = self.device)
            self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.dag_net.parameters()),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
            
        elif self.graph_type == 'dummy':
            self.dag_net = DummyDAG(self.args.num_agents, self.graph_type, self.threshold, self.device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("can only be dynamic or dummy")

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def _t2n(self, x):
        return x.detach().cpu().numpy()
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, last_actions, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.0
        """

        self.n_rollout_threads = last_actions.shape[0]

        agent_id_graph = torch.eye(self.args.num_agents).unsqueeze(0).repeat(self.n_rollout_threads, 1, 1)  # self.n_rollout_threads, num_agents, num_agents
        obs_ = check(obs).to(**self.tpdv)
        #if self.args.env_name == "GRFootball" or self.args.env_name == "gaussian":
        #    last_actions = last_actions
        #else:
        #    last_actions = np.squeeze(np.eye(self.num_actions)[last_actions.astype(np.int32)], 2)
        #last_actions_ = check(last_actions).to(**self.tpdv)
    
        obs_ = obs_.reshape(self.n_rollout_threads, self.args.num_agents, obs.shape[-1])

        #inputs_graph = torch.cat((obs_, last_actions_), -1).float()
        inputs_graph = obs_
                
        P, U, G_s, edge_noise, permutation_noise =  self.dag_net.sample(inputs_graph[:,0,:])
        
        G_s = [ig.Graph.Weighted_Adjacency(G.tolist()) for G in G_s]

        obs = obs.reshape(self.n_rollout_threads, self.args.num_agents, obs.shape[-1])

        rnn_states_actor = rnn_states_actor.reshape(self.n_rollout_threads, self.args.num_agents, rnn_states_actor.shape[-2],
                                                    rnn_states_actor.shape[-1])
        
        actions, action_log_probs, rnn_states_actor, father_actions = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks, G_s,
                                                                 available_actions,
                                                                 deterministic)
        if edge_noise!=None:
            edge_noise = edge_noise.permute(1,0,2,3).reshape(-1, 2*self.args.num_agents, self.args.num_agents)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks) # 4.1 ，  4.1.64
        #last_actions_ = last_actions_.view(-1, last_actions_.shape[-1])
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, father_actions, edge_noise, permutation_noise
        
    def get_eval_actions(self, obs, rnn_states_actor, masks, last_actions, available_actions=None,
                    deterministic=True):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.0
        """

        self.n_rollout_threads = last_actions.shape[0]

        agent_id_graph = torch.eye(self.args.num_agents).unsqueeze(0).repeat(self.n_rollout_threads, 1, 1)  # self.n_rollout_threads, num_agents, num_agents
        obs_ = check(obs).to(**self.tpdv)

        obs_ = obs_.reshape(self.n_rollout_threads, self.args.num_agents, obs.shape[-1])
        
        inputs_graph = obs_

        P, U, G_s, edge_noise, permutation_noise =  self.dag_net.sample(inputs_graph[:,0,:])
        open_ratio = 2*torch.sum(G_s)/len(G_s)/((self.args.num_agents-1)*self.args.num_agents)
        open_ratio = open_ratio.item()
        
        G_s = [ig.Graph.Weighted_Adjacency(G.tolist()) for G in G_s]
                

        obs = obs.reshape(self.n_rollout_threads, self.args.num_agents, obs.shape[-1])
        rnn_states_actor = rnn_states_actor.reshape(self.n_rollout_threads, self.args.num_agents, rnn_states_actor.shape[-2],
                                                    rnn_states_actor.shape[-1])
        actions, action_log_probs, rnn_states_actor, father_actions = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks, G_s,
                                                                 available_actions,
                                                                 deterministic)

        return actions, rnn_states_actor, open_ratio


    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        if not self.args.True_V:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        else:
            values = self.game.get_baysian_V(self, cent_obs)

        return values

    def evaluate_actions(self, cent_obs, obs, agent_id_batch, graph_obs_batch, father_action, last_actions_batch, graph_last_actions_batch, edge_noise_batch, permutation_noise_batch, rnn_states_actor, graph_rnn_states_actor, rnn_states_critic, action, masks, graph_masks,
                         available_actions=None, graph_available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        graph_rnn_states_actor = graph_rnn_states_actor.reshape(len(graph_rnn_states_actor), rnn_states_actor.shape[1], self.args.num_agents ,rnn_states_actor.shape[2])

        graph_rnn_states_actor = graph_rnn_states_actor.transpose((0,2,1,3))
        obs = check(obs).to(**self.tpdv)

        
        graph_obs_batch = check(graph_obs_batch).to(**self.tpdv)
        graph_rnn_states_actor = check(graph_rnn_states_actor).to(**self.tpdv)

        cent_obs = check(cent_obs).to(**self.tpdv)
        
        inputs_graph = graph_obs_batch
        edge_noise_batch = edge_noise_batch.permute(1,0,2,3)

        P, U, G_s, edge_noise, permutation_noise =  self.dag_net.sample(inputs_graph[:,0,:])
        
        #from here
        G_s_object = [ig.Graph.Weighted_Adjacency(G.tolist()) for G in G_s]
        
        graph_actions, graph_action_log_probs, _, graph_father_actions = self.actor(graph_obs_batch,
                                                                  graph_rnn_states_actor,
                                                                  graph_masks, G_s_object,
                                                                  available_actions = None,
                                                                  deterministic = False) #?
        
        graph_father_actions = graph_father_actions.view(len(father_action), self.args.num_agents, -1)
        
        graph_father_actions_i = []
        for gf, id in zip(graph_father_actions, agent_id_batch):
            graph_father_actions_i.append(gf[id[0],:])
        graph_father_actions_i = torch.stack(graph_father_actions_i).cuda()
        graph_father_actions_i = graph_father_actions_i.view(-1, self.args.num_agents, self.num_actions)
        graph_father_actions_i = graph_father_actions_i.detach()
        
        masked_father_action = []
        for i in range(len(father_action)):
            mask_i = G_s[i][:,agent_id_batch[i]]
            mask_i = mask_i.view(-1, 1).repeat(1, self.num_actions)
            masked_father_action.append(graph_father_actions_i[i] * mask_i)
        masked_father_action = torch.stack(masked_father_action)
        masked_father_action = masked_father_action.view(len(masked_father_action), -1)

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                      agent_id_batch,
                                                                      #father_action,
                                                                      masked_father_action,
                                                                      #edge_noise_batch,
                                                                      #permutation_noise_batch,
                                                                      rnn_states_actor,
                                                                      action,
                                                                      masks,
                                                                      available_actions,
                                                                      active_masks)
        
      
        if not self.args.True_V:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        else:
            values = None
        return values, action_log_probs, dist_entropy

    def evaluate_baysian_actions(self, obs, agent_id_batch, graph_obs_batch, father_action, last_actions_batch, edge_noise_batch, permutation_noise_batch, rnn_states_actor, graph_rnn_states_actor, rnn_states_critic, masks, graph_masks,
                         available_actions=None, graph_available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
  
        obs = check(obs).to(**self.tpdv)
        graph_obs_batch = check(graph_obs_batch).to(**self.tpdv)
        inputs_graph = [None for _ in range(2**self.args.num_agents)]
        action_probs = self.actor.evaluate_baysian_actions(obs,
                                                           agent_id_batch,
                                                           father_action,
                                                           rnn_states_actor,
                                                           masks,
                                                           available_actions,
                                                           active_masks)
        
        return action_probs
