# train_ppo_smart_buffer.py
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
import os
from smart_buffer_env_wrapper import SmartBufferObserverEnv
from observer_model import ObserverVisionTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from time import sleep
import argparse
import numpy as np


TRAIN_DATA_PATH = "results_test_new.csv"  # Path to your training dataset CSV

def make_env(env_config):
    return SmartBufferObserverEnv(env_config)

class SmartBufferCallbacks:
    """Callbacks to monitor and utilize the smart buffer."""
    
    def __init__(self):
        self.buffer_stats_history = []
    
    def on_train_result(self, algorithm, result):
        """Called after each training iteration."""
        # Get buffer statistics from environments
        env_runners = algorithm.env_runner_group
        if env_runners:
            try:
                # Get stats from first worker
                worker = env_runners.remote_workers()[0] if env_runners.remote_workers() else env_runners.local_worker()
                if worker:
                    # This would need to be implemented properly with RLlib's callback system
                    # For now, we'll log basic metrics
                    pass
            except Exception as e:
                print(f"Could not get buffer stats: {e}")
        
        # Log buffer performance
        print(f"Training iteration {result.get('training_iteration', 0)} completed")
        
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO with Smart Buffer')
    parser.add_argument('--restore-checkpoint', type=str, default=None,
                        help='Path to checkpoint directory to restore from')
    parser.add_argument('--start-iter', type=int, default=0,
                        help='Starting iteration number (useful when restoring)')
    parser.add_argument('--buffer-capacity', type=int, default=500,
                        help='Experience buffer capacity per task type')
    parser.add_argument('--task-rotation', action='store_true',
                        help='Enable task rotation for balanced sampling')
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    # Enhanced environment configuration with smart buffer
    gamma = 0.999
    env_cfg = {
        "gamma": gamma, 
        "dataset": TRAIN_DATA_PATH,
        "buffer_config": {
            "enabled": True,
            "capacity_per_task": args.buffer_capacity,
            "sample_ratio": 0.2,  # 20% from buffer, 80% fresh experiences
        },
        "task_rotation": args.task_rotation
    }

    # Register enhanced environment
    register_env("SmartBufferObserverEnv-v0", make_env)

    # GPU setup
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_gpus = 1 if visible and visible.strip() else 0

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)
        .framework("torch")
        .framework(
            eager_tracing=True,
            eager_max_retraces=20,
            tf_session_args={},
            local_tf_session_args={},
        )
        .environment(env="SmartBufferObserverEnv-v0", env_config=env_cfg)
        .rl_module(model_config=DefaultModelConfig(
                conv_filters=[
                    [16, 5, 2],
                    [32, 5, 2], 
                    [64, 3, 1], 
                    [128, 3, 1],
                    [256, 3, 1],
                ],
                conv_activation="silu",
                head_fcnet_hiddens=[256],
                vf_share_layers=True,
            )
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .training(
            gamma=gamma,
            lr=2.5e-4,
            num_epochs=10,
            train_batch_size_per_learner=576,  # matches rollout collection (12×48)
            minibatch_size=32,
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            grad_clip=10.0,
            grad_clip_by="global_norm",
        )
        .env_runners(
            num_env_runners=12,
            explore=True,
            remote_worker_envs=True,
            batch_mode="truncate_episodes",
            rollout_fragment_length=48,  # ~3-4 episodes per task
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=10,
            evaluation_num_env_runners=1,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=True,
        )
        .reporting(min_time_s_per_iteration=30)
    )

    # Build algorithm
    if args.restore_checkpoint:
        print(f"Restoring from checkpoint: {args.restore_checkpoint}")
        algo = config.build_algo()
        with torch.cuda.device(0):
            algo.restore(args.restore_checkpoint)
        print("Checkpoint restored successfully!")
    else:
        algo = config.build_algo()
        print("Starting fresh training with smart buffer...")

    # Initialize tracking
    best_eval_reward = float('-inf')
    best_checkpoint_path = None
    task_performance_history = {i: [] for i in range(4)}  # Track per-task performance

    # Load best eval reward from metadata if restoring
    if args.restore_checkpoint:
        best_metadata_file = os.path.join(os.environ['WORKING_DIR'], f"data/model_data/{TRAIN_DATA_PATH}/best_eval_metadata.txt")
        if os.path.exists(best_metadata_file):
            try:
                with open(best_metadata_file, 'r') as f:
                    best_eval_reward = float(f.read().strip())
                print(f"Loaded previous best eval reward: {best_eval_reward:.2f}")
            except:
                print("Could not load previous best eval reward, starting fresh")

    start_iter = args.start_iter
    total_iterations = 2000

    print(f"🚀 Starting training with Smart Buffer:")
    print(f"  - Buffer capacity per task: {args.buffer_capacity}")
    print(f"  - Task rotation: {args.task_rotation}")
    print(f"  - Rollout fragment length: 48")
    print(f"  - Training batch size: 576")

    for i in range(start_iter, total_iterations):
        result = algo.train()

        # Extract metrics
        ep_ret = (result.get("env_runners", {}) or {}).get("episode_return_mean")
        if ep_ret is None:
            ep_ret = result.get("episode_return_mean", float("nan"))
        
        # Evaluation metrics
        eval_reward = None
        if "evaluation" in result:
            eval_results = result['evaluation']
            eval_ep_ret = (eval_results.get("env_runners", {}) or {}).get("episode_return_mean")
            if eval_ep_ret is not None:
                eval_reward = eval_ep_ret

        # Enhanced logging
        print(f"Iter {i:4d}  train_reward={ep_ret:.2f}", end="")
        if eval_reward is not None:
            print(f"  eval_reward={eval_reward:.2f}", end="")
        else:
            print("  eval_reward=N/A", end="")
        
        # Additional smart buffer metrics
        if i % 10 == 0:  # Every 10 iterations
            print(f"  [buffer_enabled=True]", end="")
        
        print()  # New line

        # Save checkpoints
        if i % 50 == 0:
            save_dir = os.path.join(os.environ['WORKING_DIR'], f"data/model_data/{TRAIN_DATA_PATH}/ppo_smart_buffer_checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            ckpt = algo.save(save_dir)
            print(f"📁 Latest checkpoint saved to: {ckpt}")

        # Save best checkpoint
        if eval_reward is not None and eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            best_save_dir = os.path.join(os.environ['WORKING_DIR'], f"data/model_data/{TRAIN_DATA_PATH}/ppo_smart_buffer_best_checkpoint")
            os.makedirs(best_save_dir, exist_ok=True)
            best_checkpoint_path = algo.save(best_save_dir)
            print(f"🏆 New best checkpoint! Eval reward: {eval_reward:.2f} -> {best_checkpoint_path}")
            
            # Save metadata
            best_metadata_file = os.path.join(os.environ['WORKING_DIR'], f"data/model_data/{TRAIN_DATA_PATH}/best_eval_metadata.txt")
            os.makedirs(os.path.dirname(best_metadata_file), exist_ok=True)
            with open(best_metadata_file, 'w') as f:
                f.write(str(best_eval_reward))

    print(f"🎯 Training completed!")
    print(f"📊 Best evaluation reward: {best_eval_reward:.2f}")
    if best_checkpoint_path:
        print(f"💾 Best checkpoint: {best_checkpoint_path}")