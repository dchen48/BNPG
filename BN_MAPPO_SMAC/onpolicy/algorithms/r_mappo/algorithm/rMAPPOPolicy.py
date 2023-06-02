import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.r_mappo.algorithm.probabilistic_dag_model.probabilistic_dag import *
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_shape_from_obs_space
import igraph as ig

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
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.n_rollout_threads = self.args.n_rollout_threads

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.graph_type = self.args.graph_type
        self.threshold = self.args.threshold

        if self.graph_type == 'dynamic':
            graph_input_dim = get_shape_from_obs_space(self.obs_space)[0]
            self.dag_net = ProbabilisticDAG(self.args.num_agents, graph_input_dim, args.hidden_size, temperature=1.0, hard=True, noise_factor=1.0, device = self.device)
            self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.dag_net.parameters()),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)     
        elif self.graph_type == 'dummy':
            self.dag_net = DummyDAG(self.args.num_agents, self.threshold, self.device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("can only be dynamic or dummy")

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    
    def get_baysian_actions(self, obs, rnn_states_actor, masks, available_actions=None,
                    deterministic=False, test=False, evaluation = False):
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
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
                
        obs_ = check(obs).to(**self.tpdv)
        inputs_graph = obs_.reshape(-1, self.args.num_agents, obs.shape[-1])
        P, U, G_s =  self.dag_net.sample(inputs_graph, test = test)
            
        if evaluation:
            obs_alley = obs_[:,0:self.obs_space[1][0]*self.obs_space[1][1]].clone()
            obs_alley = obs_alley.view(len(obs_alley), self.obs_space[1][0], self.obs_space[1][1]).clone()
            is_alive_masks = obs_alley[:,:,0]
            ally_distance = obs_alley[:,:,1]
            ally_health = obs_alley[:,:,4]
            eval_info = {}
            for is_alive, a_d, a_h, g in zip(is_alive_masks, ally_distance, ally_health, G_s):
                for i in range(self.args.num_agents):
                    if i not in eval_info:
                        eval_info[i] = {}
                        eval_info[i]['distance'] = []
                        eval_info[i]['health'] = []
                    for j in range(self.args.num_agents-1):
                        if j>=i:
                            k=j+1
                        else:
                            k=j
                        if g[k,i]*is_alive[j] !=0:
                            eval_info[i]['distance'].append(a_d[j].item())
                            eval_info[i]['health'].append(a_h[j].item())
                        
        open_ratio = 2*torch.sum(G_s)/len(G_s)/((self.args.num_agents-1)*self.args.num_agents)
        open_ratio = open_ratio.item()
        G_s_list = [ig.Graph.Weighted_Adjacency(G.tolist()) for G in G_s]

        obs = obs.reshape(-1, self.args.num_agents, obs.shape[-1])
        rnn_states_actor = rnn_states_actor.reshape(-1, self.args.num_agents, rnn_states_actor.shape[-2],
                                                    rnn_states_actor.shape[-1])

        actions, action_log_probs, rnn_states_actor, father_actions, visible_open_ratios = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks, 
                                                                 G_s,
                                                                 G_s_list,
                                                                 available_actions,
                                                                 deterministic)
        if evaluation:
            return actions, action_log_probs, rnn_states_actor, father_actions, open_ratio, visible_open_ratios.item(), eval_info
        else:
            return actions, action_log_probs, rnn_states_actor, father_actions, open_ratio, visible_open_ratios.item()





    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
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
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
      
        actions, action_log_probs, rnn_states_actor, father_actions, _, _= self.get_baysian_actions(obs, rnn_states_actor, masks, available_actions=available_actions,
                    deterministic=deterministic, test=False)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, father_actions

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, agent_id_batch, graph_obs_batch, rnn_states_actor, graph_rnn_states_actor, rnn_states_critic, action, father_action, masks, graph_masks,
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
        
        graph_available_actions = graph_available_actions.reshape(len(graph_available_actions), self.args.num_agents, available_actions.shape[1])
        
        #if self.args.use_recurrent_policy:
        #    graph_rnn_states_actor = graph_rnn_states_actor.reshape(len(graph_rnn_states_actor), rnn_states_actor.shape[1], self.args.num_agents ,rnn_states_actor.shape[2])
        #    graph_rnn_states_actor = graph_rnn_states_actor.transpose((0,2,1,3))
        #    graph_rnn_states_actor = check(graph_rnn_states_actor).to(**self.tpdv)

        graph_obs_batch = check(graph_obs_batch).to(**self.tpdv)
        
        
        P, U, G_s =  self.dag_net.sample(graph_obs_batch, test = False)   
        open_ratio = 2*torch.sum(G_s)/len(G_s)/((self.args.num_agents-1)*self.args.num_agents)     
        G_s_list = [ig.Graph.Weighted_Adjacency(G.tolist()) for G in G_s]

       
        action_log_probs, dist_entropy = self.actor.evaluate_actions(agent_id_batch,
                                                                     obs, 
                                                                     rnn_states_actor,
                                                                     action,
                                                                     father_action,
                                                                     masks,
                                                                     G_s,
                                                                     G_s_list,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, open_ratio

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        actions, _, rnn_states_actor, _, open_ratio, visible_open_ratios, eval_info = self.get_baysian_actions(obs, rnn_states_actor, masks, available_actions=available_actions,
                    deterministic=deterministic, test=False, evaluation = True) #test = False??
        return actions, rnn_states_actor, open_ratio, visible_open_ratios, eval_info
