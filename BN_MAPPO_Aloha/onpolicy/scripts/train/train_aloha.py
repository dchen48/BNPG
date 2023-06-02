#!/usr/bin/env python
import sys
import os
import socket
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.aloha.aloha import AlohaEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from datetime import datetime, timedelta

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = AlohaEnv(n_agents=all_args.num_agents,
                           episode_limit=all_args.episode_length,
                           max_list_length=5,
                           obs_last_action=False,
                           state_last_action=True,
                           seed=all_args.seed)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = AlohaEnv(n_agents=all_args.num_agents,
                           episode_limit=all_args.episode_length,
                           max_list_length=5,
                           obs_last_action=False,
                           state_last_action=True,
                           seed=all_args.seed)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int, default=10, help="number of players")
    
    all_args = parser.parse_known_args(args)[0]
    
    all_args.env_name = "Aloha"
    all_args.n_rollout_threads = 32
    all_args.n_training_threads = 32
    all_args.episode_length = 20
    all_args.num_env_steps = 1000000
    all_args.ppo_epoch = 5
    all_args.lr = 7e-4
    all_args.critic_lr = 7e-4

    all_args.eval_interval = 25
    all_args.eval_episodes = 100
    
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
    }

    # run experiments
    from onpolicy.runner.shared.aloha_runner import AlohaRunner as Runner
    
    runner = Runner(config)
    runner.run()
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])
