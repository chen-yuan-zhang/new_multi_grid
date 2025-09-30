# train_ppo.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import os
from observer_env_wrapper import ObserverEnvDirectRew
from observer_model import ObserverVisionTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

def make_env(env_config):
    return ObserverEnvDirectRew(env_config)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # Env registration (string name is convenient for configs)
    register_env("ObserverEnv-v0", make_env)

    gamma = 0.995

    # Goes into your ObserverEnv
    env_cfg = {"gamma": gamma, "dataset": "results.csv"}

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
            train_batch_size_per_learner=750,  # samples aggregated per update
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

    algo = config.build_algo()

    for i in range(400):
        result = algo.train()

        # New-stack metrics live under result["env_runners"]
        ep_ret = (result.get("env_runners", {}) or {}).get("episode_return_mean")
        if ep_ret is None:
            ep_ret = result.get("episode_return_mean", float("nan"))
        print(f"Iter {i}  episode_return_mean={ep_ret:.2f}")       # :contentReference[oaicite:5]{index=5}

        if i % 50 == 0:
            save_dir = "/home/sukai/Project/chenyuan_project/new_multi_grid_new_rl/multigrid/pure_rl/model_data/ppo_observer_checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            ckpt = algo.save(save_dir)
            print("Checkpoint saved to:", ckpt)