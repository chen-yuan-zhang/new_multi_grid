#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/home/sukaih/Extrastorage/new_multi_grid_new_rl
export WORKING_DIR=/home/sukaih/Extrastorage/new_multi_grid_new_rl
# export ACTOR_PREDICTOR_MODEL_PATH=/home/sukai/Project/chenyuan_project/outputs/unsloth_finetune
export RL_POLICY_CHECKPOINT_PATH=/home/sukaih/Extrastorage/new_multi_grid_new_rl/data/model_data/formal_dataset_v0.csv/ppo_observer_checkpoints
export WANDB_DIR=/home/sukaih/wandb 
export WANDB_PROJECT=chenyuan_action_prediction
export HF_HOME=/home/sukaih/hf_cache
export TRITON_CACHE_DIR=/home/sukaih/triton_cache
export CUDA_VISIBLE_DEVICES=0
ulimit -n 999999 # Increase open file limit
