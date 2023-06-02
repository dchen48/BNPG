import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import os
import json

import sys

def _t2n(x):
    return x.detach().cpu().numpy()

class AlohaRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(AlohaRunner, self).__init__(config)

    def run(self):
        self.warmup()
    
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        save_interval = max(int(episodes/20), 1)
        eval_episodic_rewards = {}
        eval_episodic_open_ratios = {}
        for episode in range(episodes):
            if episode % 20 == 0:
                print('%training finished: ',episode/episodes)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, father_actions, edge_noise, permutation_noise = self.collect(step)  ##
               
           
                obs, rewards, dones, infos = self.envs.step(actions_env)
                
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, father_actions, edge_noise, permutation_noise
              
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % save_interval == 0 or episode == episodes - 1):
                self.save(total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                avg_episodic_rewards, avg_open_ratios = self.eval()
                eval_episodic_rewards[total_num_steps] = avg_episodic_rewards
                eval_episodic_open_ratios[total_num_steps] = avg_open_ratios
        
        if not os.path.exists(self.save_dir+ '/eval_episodic_reward/'):
            os.makedirs(self.save_dir+ '/eval_episodic_reward/')
        if not os.path.exists(self.save_dir+ '/eval_episodic_open_ratios/'):
            os.makedirs(self.save_dir+ '/eval_episodic_open_ratios/')

        with open(self.save_dir + '/eval_episodic_reward/'+str(self.all_args.seed)+'.json', 'w') as fp:
            json.dump(eval_episodic_rewards, fp)
        
        with open(self.save_dir + '/eval_episodic_open_ratios/'+str(self.all_args.seed)+'.json', 'w') as fp:
            json.dump(eval_episodic_open_ratios, fp)
      
    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()  # change mode to prepare the rollout
        value, action, action_log_prob, rnn_states, rnn_states_critic, father_actions, edge_noise, permutation_noise \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),  ## 4.1.96 --> 4.96
                            np.concatenate(self.buffer.obs[step]), # 1.4.24  --> 4.24
                            np.concatenate(self.buffer.rnn_states[step]), # 1.4.1.64  --> 4.1.64
                            np.concatenate(self.buffer.rnn_states_critic[step]),  # 1.4.1.64 --> 4.1.64
                            np.concatenate(self.buffer.masks[step]),
                            self.buffer.actions[step])
                            # np.concatenate(self.buffer.actions[step]) )  # 1.4.1 --> 4.1
      
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        #rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states = None
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        father_actions = np.array(np.split(_t2n(father_actions), self.n_rollout_threads))
        if edge_noise!=None:
            edge_noise = _t2n(edge_noise)
        permutation_noise = _t2n(permutation_noise)
        #last_actions = np.array(np.split(_t2n(last_actions), self.n_rollout_threads))
        
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2) # 1.4.5
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, father_actions, edge_noise, permutation_noise

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, father_actions, edge_noise, permutation_noise = data

        #rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)     #threads * n * n * obs_dim
        else:
            share_obs = obs
                    
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, father_actions, edge_noise, permutation_noise)

    @torch.no_grad()
    def eval(self):
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.trainer.prep_rollout()
        last_actions = np.zeros((self.n_eval_rollout_threads, self.all_args.num_agents, 1), dtype=np.float32)
        avg_episodic_rewards = []
        avg_open_ratios = []
        
        for _ in range(self.all_args.eval_episodes):
            open_ratios = []
            eval_episode_rewards = []
            for eval_step in range(self.episode_length):
                eval_action, eval_rnn_states, open_ratio = self.trainer.policy.get_eval_actions(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    last_actions,
                                                    deterministic=True)
                open_ratios.append(open_ratio)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                #eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                
                if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[0].shape):
                        eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                        if i == 0:
                            eval_actions_env = eval_uc_actions_env
                        else:
                            eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
                elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                    eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                eval_episode_rewards.append(eval_rewards)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            open_ratios = np.mean(open_ratios)
            avg_open_ratios.append(open_ratios)
            eval_episode_rewards = np.array(eval_episode_rewards)
            eval_episode_rewards = np.mean(np.sum(np.array(eval_episode_rewards), axis=0))
          
            avg_episodic_rewards.append(eval_episode_rewards)
        return np.mean(avg_episodic_rewards), np.mean(avg_open_ratios)


    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array', close=False)[0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array', close=False)[0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(ifi - elapsed)

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + 'render.gif', all_frames, duration=self.all_args.ifi)
