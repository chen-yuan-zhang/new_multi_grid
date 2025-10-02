# train_ppo.py
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
import os
from observer_env_wrapper import ObserverEnvDirectRew
from observer_model import ObserverVisionTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from time import sleep

TRAIN_DATA_PATH = "formal_dataset_v0.csv"  # Path to your training dataset CSV
import argparse
def make_env(env_config):
    return ObserverEnvDirectRew(env_config)

if __name__ == "__main__":

    # Add command line argument parsing for checkpoint restoration
    parser = argparse.ArgumentParser(description='Train PPO with optional checkpoint restoration')
    parser.add_argument('--restore-checkpoint', type=str, default=None,
                        help='Path to checkpoint directory to restore from')
    parser.add_argument('--start-iter', type=int, default=0,
                        help='Starting iteration number (useful when restoring)')
    args = parser.parse_args()


    ray.init(ignore_reinit_error=True)

    # Env registration (string name is convenient for configs)
    register_env("ObserverEnv-v0", make_env)

    gamma = 0.999

    # Goes into your ObserverEnv
    env_cfg = {"gamma": gamma, "dataset": TRAIN_DATA_PATH}

    # Simple GPU selector: use 1 GPU if any are visible
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_gpus = 1 if visible and visible.strip() else 0

    config = (
        PPOConfig()
        # Ensure the NEW stack is on (RLModule/Learner + EnvRunners/ConnectorV2)
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)     # :contentReference[oaicite:0]{index=0}
        .framework("torch")
        .framework(
            eager_tracing=True,
            eager_max_retraces=20,
            tf_session_args={},
            local_tf_session_args={},
        )
        .environment(env="ObserverEnv-v0", env_config=env_cfg)
        .rl_module(model_config=DefaultModelConfig(
                conv_filters=[
                    [16, 5, 2],
                    [32, 5, 2], 
                    [64, 3, 1], 
                    [128, 3, 1],
                    [256, 3, 1],
                ],
                conv_activation="silu",
                # After the last CNN, the default model flattens, then adds an optional MLP.
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
            num_epochs=10,               # passes over the train batch
            train_batch_size_per_learner=2000,  # samples aggregated per update
            minibatch_size=32,
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            grad_clip=100.0,
            grad_clip_by="global_norm",
        )                                                          # :contentReference[oaicite:2]{index=2}
        .env_runners(
            num_env_runners=12,
            explore=True,
            remote_worker_envs=True,
            batch_mode="complete_episodes",
            
        )                                                          # :contentReference
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=10,                  # episodes per eval
            evaluation_num_env_runners=1,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=True,
            
        )                                                         
        .reporting(min_time_s_per_iteration=30)
    )

    # Build or restore algorithm
    if args.restore_checkpoint:
        print(f"Restoring from checkpoint: {args.restore_checkpoint}")
        algo = config.build_algo()
        with torch.cuda.device(0):
            algo.restore(args.restore_checkpoint)

        print("Checkpoint restored successfully!")
        
    else:
        algo = config.build_algo()
        print("Starting fresh training...")

    

    # Track best evaluation performance
    best_eval_reward = float('-inf')
    best_checkpoint_path = None

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
    total_iterations = 600

    for i in range(start_iter, total_iterations):
        result = algo.train()

        # New-stack metrics live under result["env_runners"]
        ep_ret = (result.get("env_runners", {}) or {}).get("episode_return_mean")
        if ep_ret is None:
            ep_ret = result.get("episode_return_mean", float("nan"))
        
        # Get evaluation metrics
        eval_reward = None
        if "evaluation" in result:
            eval_reward = result["evaluation"].get("episode_return_mean")
        
        print(f"Iter {i}  episode_return_mean={ep_ret:.2f}", end="")
        if eval_reward is not None:
            print(f"  eval_reward={eval_reward:.2f}")
        else:
            print()

        # Save latest checkpoint every 50 iterations
        if i % 50 == 0:
            save_dir = os.path.join(os.environ['WORKING_DIR'], f"data/model_data/{TRAIN_DATA_PATH}/ppo_observer_checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            ckpt = algo.save(save_dir)
            print("Latest checkpoint saved to:", ckpt)

        # Save best checkpoint based on evaluation results
        if eval_reward is not None and eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            best_save_dir = os.path.join(os.environ['WORKING_DIR'], f"data/model_data/{TRAIN_DATA_PATH}/ppo_observer_best_checkpoint")
            os.makedirs(best_save_dir, exist_ok=True)
            best_checkpoint_path = algo.save(best_save_dir)
            print(f"New best checkpoint saved! Eval reward: {eval_reward:.2f} -> {best_checkpoint_path}")
            
            # Save best eval reward metadata for future restores
            best_metadata_file = os.path.join(os.environ['WORKING_DIR'], f"data/model_data/{TRAIN_DATA_PATH}/best_eval_metadata.txt")
            os.makedirs(os.path.dirname(best_metadata_file), exist_ok=True)
            with open(best_metadata_file, 'w') as f:
                f.write(str(best_eval_reward))

    print(f"Training completed. Best evaluation reward: {best_eval_reward:.2f}")
    if best_checkpoint_path:
        print(f"Best checkpoint saved at: {best_checkpoint_path}")