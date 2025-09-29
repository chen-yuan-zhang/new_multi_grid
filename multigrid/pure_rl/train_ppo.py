# train_ppo.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import os
from observer_env_wrapper import ObserverEnv
from observer_model import ObserverVisionTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

def make_env(env_config):
    return ObserverEnv(env_config)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # Env registration (string name is convenient for configs)
    register_env("ObserverEnv-v0", make_env)

    gamma = 0.995

    # ---- RLModule spec (replaces ModelV2/custom_model) ----
    rlm_spec = RLModuleSpec(
        module_class=ObserverVisionTorchRLModule,
        model_config={
            # ConnectorV2 uses this to pad/truncate RNN sequences (TBPTT)
            "max_seq_len": 32,
            "custom_model_config": {
                "use_gru": True,
                "gru_hidden": 256,
                "gru_layers": 1,
                "fc_hidden": 512,
                "frame_norm": True,
                "channels_last": False,
                "target_spatial": 7,
            },
        },
    )

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
        .environment(env="ObserverEnv-v0", env_config=env_cfg)
        .rl_module(rl_module_spec=rlm_spec)                      # :contentReference[oaicite:1]{index=1}
        .training(
            gamma=gamma,
            lr=2.5e-4,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            grad_clip=0.5,
            # New knob names on the new stack:
            minibatch_size=512,          # per optimizer step on GPU
            num_epochs=10,               # passes over the train batch
            train_batch_size_per_learner=8192,  # samples aggregated per update
        )                                                          # :contentReference[oaicite:2]{index=2}
        .env_runners(
            num_env_runners=3,
            num_envs_per_env_runner=1,
            rollout_fragment_length=64,  # multiple of max_seq_len=32
            batch_mode="truncate_episodes",
        )                                                          # :contentReference[oaicite:3]{index=3}
        .resources(num_gpus=num_gpus)
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=10,                  # episodes per eval
            evaluation_num_env_runners=1,
            evaluation_parallel_to_training=False,
            evaluation_config={"explore": False},
        )                                                          # :contentReference[oaicite:4]{index=4}
        .reporting(min_time_s_per_iteration=30)
    )

    algo = config.build()

    for i in range(1000):
        result = algo.train()

        # New-stack metrics live under result["env_runners"]
        ep_ret = (result.get("env_runners", {}) or {}).get("episode_return_mean")
        if ep_ret is None:
            ep_ret = result.get("episode_return_mean", float("nan"))
        print(f"Iter {i}  episode_return_mean={ep_ret:.2f}")       # :contentReference[oaicite:5]{index=5}

        if i % 50 == 0:
            ckpt = algo.save()
            print("Checkpoint saved to:", ckpt)