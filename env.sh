#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/home/sukai/Project/chenyuan_project/new_multi_grid_new_rl
export WORKING_DIR=/home/sukai/Project/chenyuan_project/new_multi_grid_new_rl
export ACTOR_PREDICTOR_MODEL_PATH=/home/sukai/Project/chenyuan_project/outputs/unsloth_finetune
export RL_POLICY_CHECKPOINT_PATH=/home/sukai/Project/chenyuan_project/new_multi_grid_new_rl/multigrid/pure_rl/model_data/ppo_observer_checkpoints
export WANDB_DIR=/home/sukai/wandb 
export WANDB_PROJECT=chenyuan_action_prediction
export HF_HOME=/home/sukai/hf_cache
export TRITON_CACHE_DIR=/home/sukai/triton_cache 
export CUDA_VISIBLE_DEVICES=0
ulimit -n 999999 # Increase open file limit
