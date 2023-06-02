import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner
import os
import json

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
        self.threshold = 1
        self.alpha = self.all_args.alpha
       
    def run(self):
        self.warmup()   
        if self.all_args.map_name == '6h_vs_8z':
            initial_alpha = 1
        else:
            initial_alpha = 0.5
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        eval_win_rates = {}
        eval_open_ratios = {}
        eval_visible_open_ratios = {}
        eval_rewards = {}
        eval_infos = {}
        save_interval = max(int(episodes/20), 1)
        decay_interval = max(int(episodes*0.6/self.num_decay/2),1)
        alpha_increase_interval = max(int(episodes*0.2/self.num_decay/2),1)
     
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, father_actions = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, father_actions 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train(self.threshold, self.all_args.alpha)
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % save_interval == 0 or episode == episodes - 1):
                self.save(total_num_steps)
            
            if (episode/episodes>=0.4 and episode % decay_interval == 0):
                self.threshold = max(self.threshold-1/self.num_decay,0)
            if (episode/episodes>=0.7 and episode % alpha_increase_interval == 0):
                self.all_args.alpha = min(self.all_args.alpha+(initial_alpha-self.alpha)/alpha_increase_interval,1)

            # eval
            if episode % self.eval_interval == 0:
                eval_win_rate, eval_open_ratio, eval_reward, eval_visible_open_ratio, eval_ret_info = self.eval()
                eval_win_rates[total_num_steps] = eval_win_rate
                eval_open_ratios[total_num_steps] = eval_open_ratio
                eval_visible_open_ratios[total_num_steps] = eval_visible_open_ratio
                eval_rewards[total_num_steps] = eval_reward
                eval_ret_info['threshold'] = self.threshold
                eval_infos[total_num_steps] = eval_ret_info
        
        if not os.path.exists(self.save_dir+ '/eval_win_rate/'):
            os.makedirs(self.save_dir+ '/eval_win_rate/')
        if not os.path.exists(self.save_dir+ '/eval_open_ratio/'):
            os.makedirs(self.save_dir+ '/eval_open_ratio/')
        if not os.path.exists(self.save_dir+ '/eval_visible_open_ratio/'):
            os.makedirs(self.save_dir+ '/eval_visible_open_ratio/')
        if not os.path.exists(self.save_dir+ '/eval_reward/'):
            os.makedirs(self.save_dir+ '/eval_reward/')
        if not os.path.exists(self.save_dir+ '/eval_info/'):
            os.makedirs(self.save_dir+ '/eval_info/')
        

        with open(self.save_dir + '/eval_win_rate/'+str(self.all_args.seed)+'.json', 'w') as fp:
            json.dump(eval_win_rates, fp)
        
        with open(self.save_dir + '/eval_open_ratio/'+str(self.all_args.seed)+'.json', 'w') as fp:
            json.dump(eval_open_ratios, fp)
        
        with open(self.save_dir + '/eval_visible_open_ratio/'+str(self.all_args.seed)+'.json', 'w') as fp:
            json.dump(eval_visible_open_ratios, fp)

        with open(self.save_dir + '/eval_reward/'+str(self.all_args.seed)+'.json', 'w') as fp:
            json.dump(eval_rewards, fp)
        
        with open(self.save_dir + '/eval_info/'+str(self.all_args.seed)+'.json', 'w') as fp:
            json.dump(eval_infos, fp)
        
    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic, father_actions\
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        father_actions = np.array(np.split(_t2n(father_actions), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, father_actions

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, father_actions = data

        dones_env = np.all(dones, axis=1)
        
        if self.all_args.use_recurrent_policy:
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, father_actions, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, threshold=0):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        eval_open_ratio = []
        eval_visible_open_ratio = []
        eval_distance = {}
        eval_health = {}
        for i in range(self.all_args.num_agents):
            eval_distance[i] = []
            eval_health[i] = []
        
        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states, open_ratio, visible_open_ratio, eval_info = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            for i in range(self.all_args.num_agents):
                eval_distance[i] += eval_info[i]['distance']
                eval_health[i] += eval_info[i]['health']
          
            eval_open_ratio.append(open_ratio)
            eval_visible_open_ratio.append(visible_open_ratio)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)
            
            eval_dones_env = np.all(eval_dones, axis=1)
            
            if self.all_args.use_recurrent_policy:
                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = torch.tensor(eval_episode_rewards)
                eval_episode_rewards = eval_episode_rewards.view(self.all_args.eval_episodes, -1)
                eval_episode_rewards = torch.mean(eval_episode_rewards, dim=1)
                eval_episode_rewards = torch.mean(eval_episode_rewards)
                ret_info = {}
                for i in range(self.all_args.num_agents):
                    if eval_distance[i]!=[]:
                        eval_distance[i] = np.mean(eval_distance[i])
                        eval_health[i] = np.mean(eval_health[i])
                ret_info['distance'] = eval_distance
                ret_info['health'] = eval_health

                #eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                #self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                #print("eval win rate is {}.".format(eval_win_rate))
                return eval_win_rate, np.mean(eval_open_ratio), eval_episode_rewards.item(), np.mean(eval_visible_open_ratio), ret_info
    

    def eval_decentralize(self):
        avg_win_rate = []
        avg_reward = []
        for seed in range(1,6):
            model_dir = '/content/gdrive/MyDrive/new/parent_replay/6h_vs_8z/state_last_action_False/obs_last_action_False/dynamic/decay/20/weights/'+str(seed)
            eval_win_rate, eval_open_ratio, eval_episode_rewards, eval_visible_open_ratio, ret_info = self.eval_decentralize_helper(model_dir)
            print('eval_win_rate: ',eval_win_rate)
            avg_win_rate.append(eval_win_rate)
            avg_reward.append(eval_episode_rewards)
            
        avg_win_rate = np.mean(avg_win_rate)
        avg_reward = np.mean(avg_reward)
        print('avg_win_rate: ',avg_win_rate)
        print('avg_reward: ',avg_reward)
        return avg_win_rate, avg_reward

    @torch.no_grad()
    def eval_decentralize_helper(self, model_dir):
        self.restore(model_dir)
        
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        eval_open_ratio = []
        eval_visible_open_ratio = []
        eval_distance = {}
        eval_health = {}
        for i in range(self.all_args.num_agents):
            eval_distance[i] = []
            eval_health[i] = []
        
        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states, open_ratio, visible_open_ratio, eval_info = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            for i in range(self.all_args.num_agents):
                eval_distance[i] += eval_info[i]['distance']
                eval_health[i] += eval_info[i]['health']
          
            eval_open_ratio.append(open_ratio)
            eval_visible_open_ratio.append(visible_open_ratio)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)
            
            eval_dones_env = np.all(eval_dones, axis=1)
            
            if self.all_args.use_recurrent_policy:
                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = torch.tensor(eval_episode_rewards)
                eval_episode_rewards = eval_episode_rewards.view(self.all_args.eval_episodes, -1)
                eval_episode_rewards = torch.mean(eval_episode_rewards, dim=1)
                eval_episode_rewards = torch.mean(eval_episode_rewards)
                ret_info = {}
                for i in range(self.all_args.num_agents):
                    if eval_distance[i]!=[]:
                        eval_distance[i] = np.mean(eval_distance[i])
                        eval_health[i] = np.mean(eval_health[i])
                ret_info['distance'] = eval_distance
                ret_info['health'] = eval_health

                #eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                #self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                #print("eval win rate is {}.".format(eval_win_rate))
                return eval_win_rate, np.mean(eval_open_ratio), eval_episode_rewards.item(), np.mean(eval_visible_open_ratio), ret_info
                
