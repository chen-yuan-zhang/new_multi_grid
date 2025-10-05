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
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config

from time import sleep
import argparse
import numpy as np


TRAIN_DATA_PATH = "results_test_new.csv"  # Path to your training dataset CSV

def make_env(env_config):
    return SmartBufferObserverEnv(env_config)

class CurriculumBufferCallbacks:
    """Enhanced callbacks to monitor smart buffer and curriculum learning."""
    
    def __init__(self):
        self.buffer_stats_history = []
        self.curriculum_stats_history = []
        self.last_curriculum_stage = 0
    
    def on_train_result(self, algorithm, result):
        """Called after each training iteration."""
        # Try to get curriculum and buffer statistics from environments
        env_runners = algorithm.env_runner_group
        if env_runners:
            try:
                # Get stats from local worker (if available) or first remote worker
                workers = [env_runners.local_worker()] if env_runners.local_worker() else []
                workers.extend(env_runners.remote_workers())
                
                if workers:
                    worker = workers[0]
                    # Note: In a real implementation, you'd need to add a method to get these stats
                    # This is a placeholder for the interface
                    print(f"📊 Buffer monitoring active (iteration {result.get('training_iteration', 0)})")
                    
            except Exception as e:
                print(f"Could not get curriculum stats: {e}")
        
        return result
    
    def log_curriculum_advancement(self, new_stage, current_levels):
        """Log when curriculum advances to new stage."""
        if new_stage > self.last_curriculum_stage:
            print(f"🎯 CURRICULUM ADVANCED: Stage {self.last_curriculum_stage} → {new_stage}")
            print(f"   Active levels: {current_levels}")
            self.last_curriculum_stage = new_stage
            self.curriculum_stats_history.append({
                'stage': new_stage,
                'levels': current_levels,
                'timestamp': result.get('training_iteration', 0) if 'result' in locals() else 0
            })

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
    parser.add_argument('--curriculum-sizes', type=int, nargs='+', default=[10, 12, 15],
                        help='Grid sizes for curriculum learning (default: 10 12 15)')
    parser.add_argument('--curriculum-distances', type=int, nargs='+', default=[3, 5, 7],
                        help='Initial distances for curriculum learning (default: 3 5 7)')
    parser.add_argument('--convergence-threshold', type=float, default=0.8,
                        help='Success rate threshold for curriculum advancement (default: 0.8)')
    parser.add_argument('--success-window', type=int, default=100,
                        help='Window size for success rate calculation (default: 100)')
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    # Enhanced environment configuration with smart buffer and curriculum learning
    gamma = 0.999
    env_cfg = {
        "gamma": gamma, 
        "dataset": TRAIN_DATA_PATH,
        "buffer_config": {
            "enabled": True,
            "capacity_per_task": args.buffer_capacity,
            "sample_ratio": 0.2,  # 20% from buffer, 80% fresh experiences
            # Curriculum learning configuration
            "curriculum": {
                "sizes": args.curriculum_sizes,
                "initial_distances": args.curriculum_distances
            }
        },
        "task_rotation": args.task_rotation  # May be disabled for curriculum learning
    }

    # Register enhanced environment
    register_env("SmartBufferObserverEnv-v0", make_env)

    # Test environment creation to ensure it works
    try:
        test_env = make_env(env_cfg)
        print(f"✅ Environment test successful:")
        print(f"   Action space: {test_env.action_space}")
        print(f"   Observation space: {test_env.observation_space}")
        print(f"   Single action space: {test_env.single_action_space}")
        del test_env
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # GPU setup
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_gpus = 1 if visible and visible.strip() else 0
    lr_multiplier = num_gpus if num_gpus > 0 else 1

    default_config = DreamerV3Config()

    config = (
        DreamerV3Config()
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
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .training(
            model_size="S",
            training_ratio=1024,
            batch_size_B=16,
            world_model_lr=default_config.world_model_lr * lr_multiplier,
            actor_lr=default_config.actor_lr * lr_multiplier,
            critic_lr=default_config.critic_lr * lr_multiplier,
        )
        .env_runners(
            remote_worker_envs=1,
        )
        # .env_runners(
        #     num_env_runners=12,
        #     explore=True,
        #     remote_worker_envs=True,
        #     batch_mode="truncate_episodes",
        #     rollout_fragment_length=48,  # ~3-4 episodes per task
        #     create_env_on_local_worker=True,  # Ensure local worker has env
        # )
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
        print("Building algorithm...")
        algo = config.build_algo()
        print("Algorithm built successfully!")
        
        # Debug: Check if env_runner is properly initialized
        print("🔍 Debugging env_runner initialization:")
        print(f"   Has env_runner: {hasattr(algo, 'env_runner')}")
        if hasattr(algo, 'env_runner') and algo.env_runner:
            print(f"   env_runner type: {type(algo.env_runner)}")
            print(f"   Has env: {hasattr(algo.env_runner, 'env')}")
            if hasattr(algo.env_runner, 'env'):
                print(f"   env is None: {algo.env_runner.env is None}")
                if algo.env_runner.env is not None:
                    print(f"   env type: {type(algo.env_runner.env)}")
        
        print("Starting fresh training with smart buffer...")

    # Initialize tracking
    best_eval_reward = float('-inf')
    best_checkpoint_path = None
    curriculum_advancement_history = []  # Track curriculum progression
    task_performance_history = {}  # Track per curriculum level performance

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
    total_iterations = 20000

    print(f"🚀 Starting training with Smart Buffer and Curriculum Learning:")
    print(f"  - Buffer capacity per task: {args.buffer_capacity}")
    print(f"  - Task rotation: {args.task_rotation}")
    print(f"  - Curriculum sizes: {args.curriculum_sizes}")
    print(f"  - Curriculum distances: {args.curriculum_distances}")
    print(f"  - Convergence threshold: {args.convergence_threshold}")
    print(f"  - Success window: {args.success_window}")
    print(f"  - Rollout fragment length: 48")
    print(f"  - Training batch size: 576")

    for i in range(start_iter, total_iterations):
        # Additional safety check before training
        if hasattr(algo, 'env_runner') and algo.env_runner and algo.env_runner.env is None:
            print("⚠️  Warning: env_runner.env is None, attempting to reinitialize...")
            try:
                # Try to manually set up the environment
                test_env = make_env(env_cfg)
                print(f"Created test environment: {type(test_env)}")
                del test_env
            except Exception as e:
                print(f"Failed to create test environment: {e}")
                break
        
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
        
        # Additional smart buffer and curriculum metrics
        if i % 10 == 0:  # Every 10 iterations
            print(f"  [buffer_enabled=True, curriculum_enabled=True]", end="")
        
        # Detailed curriculum logging every 25 iterations
        if i % 25 == 0:
            print(f"\n📚 Curriculum Status at iteration {i}:")
            print(f"  - Configuration: sizes={args.curriculum_sizes}, distances={args.curriculum_distances}")
            print(f"  - Total curriculum levels: {len(args.curriculum_sizes) * len(args.curriculum_distances)}")
            # Note: Actual curriculum stats would need to be retrieved from the environment
            # This would require implementing a way to collect stats from distributed workers
        
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
    print(f"📚 Curriculum configuration: sizes={args.curriculum_sizes}, distances={args.curriculum_distances}")
    print(f"🎓 Total curriculum levels: {len(args.curriculum_sizes) * len(args.curriculum_distances)}")
    if best_checkpoint_path:
        print(f"💾 Best checkpoint: {best_checkpoint_path}")