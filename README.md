# BNPG

## Training
1.  Train fixed DAG topologies with tabular exact policy gradients on Coordination Game

```
cd Tabular_Coordination_Game
python run.py
```

2. Train context-aware DAG topologies with MAPPO on Coordination Game

```
cd BN_MAPPO_Coordination_Game/onpolicy/scripts/train
python train_coordination_game.py --graph_type [dummy, dynamic] --threshold [0,1]
```

3. Train context-aware DAG topologies with MAPPO on Aloha

```
cd BN_MAPPO_Aloha/onpolicy/scripts/train
python train_aloha.py --graph_type [dummy, dynamic] --threshold [0,1]
```

4. Train context-aware DAG topologies with MAPPO on SMAC

```
cd BN_MAPPO_SMAC/onpolicy/scripts/train

#on 6h_vs_8z, --use_annealing for annealing strategy
python train_aloha.py train_smac.py --use_annealing --graph_type [dummy, dynamic] --threshold [0,1] \ 
--alpha 0.1 --env_name StarCraft2 --algorithm_name mappo --use_recurrent_policy \
--map_name 6h_vs_8z --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
 --num_env_steps 20000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 

#on MMM2, --use_annealing for annealing strategy
python train_aloha.py train_smac.py --use_annealing --graph_type [dummy, dynamic] --threshold [0,1] \
--alpha 0.05 --env_name StarCraft2 --algorithm_name mappo --use_recurrent_policy \
    --map_name MMM2 --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 2 --episode_length 400 \
    --num_env_steps 20000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 
```



### Acknowledgement
The MAPPO code is based on https://github.com/marlbenchmark/on-policy

The differentiable DAG sampling code is based on https://github.com/sharpenb/Differentiable-DAG-Sampling
