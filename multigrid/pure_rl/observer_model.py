from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import (
    TorchCategorical,
    TorchDiagGaussian,
)
from ray.rllib.utils.annotations import override


class ObserverVisionTorchRLModule(TorchRLModule, ValueFunctionAPI):
    """
    CNN policy/value with optional GRU memory (new RLModule API).

    Expects image observations; supported shapes at runtime:
      - [B, C, H, W] or [B, H, W, C] (channels_last=True)
      - [B, T, C, H, W] or [B, T, H, W, C]

    model_config_dict may include:
      custom_model_config:
        use_gru: bool = True
        gru_hidden: int = 256
        gru_layers: int = 1
        fc_hidden: int = 512
        frame_norm: bool = True        # if uint8 frames, scales to [0,1]
        channels_last: bool = False    # set True if obs is (H,W,C)
        target_spatial: int = 7        # adaptive pool target (H=W)
      max_seq_len: int = 32            # used by ConnectorV2 for TBPTT
    """

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config: Optional[Dict[str, Any]] = None,
        inference_only: bool = False,
        catalog_class=None,  # unused; kept for compatibility with RLModuleSpec
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            model_config=model_config,
        )

        # Read config (support dict or DefaultModelConfig-like objects).
        cfg_all = model_config or {}
        if hasattr(cfg_all, "to_dict"):  # e.g., DefaultModelConfig
            cfg_all = cfg_all.to_dict()
        custom = cfg_all.get("custom_model_config", {}) or {}

        self.use_gru: bool = bool(custom.get("use_gru", True))
        self.gru_hidden: int = int(custom.get("gru_hidden", 256))
        self.gru_layers: int = int(custom.get("gru_layers", 1))
        self.fc_hidden: int = int(custom.get("fc_hidden", 512))
        self.frame_norm: bool = bool(custom.get("frame_norm", True))
        self.channels_last: bool = bool(custom.get("channels_last", False))
        self.target_spatial: int = int(custom.get("target_spatial", 7))

        # Build the actual networks in setup() (called by RLlib).
        # We only cache some shape hints here.
        shp = observation_space.shape
        if shp is None or len(shp) < 3:
            raise ValueError("ObserverVisionTorchRLModule expects 3D+ image observations.")
        if self.channels_last:
            H, W, C = shp[-3], shp[-2], shp[-1]
        else:
            C, H, W = shp[-3], shp[-2], shp[-1]
        self._obs_C = int(C)
        self._obs_H = int(H)
        self._obs_W = int(W)

        # Pick an action dist class compatible with the space.
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dist_cls = TorchCategorical
            self._num_outputs = int(action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            # Diagonal Gaussian: mean+log_std for each dim
            self.action_dist_cls = TorchDiagGaussian
            self._num_outputs = int(action_space.shape[0]) * 2
        else:
            raise NotImplementedError(f"Unsupported action space: {action_space}")

    @override(TorchRLModule)
    def setup(self) -> None:
        # ---- CNN encoder (Nature DQN style) ----
        self.conv = nn.Sequential(
            nn.Conv2d(self._obs_C, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_spatial, self.target_spatial))

        # Probe conv output dimensionality with a dummy tensor.
        with torch.no_grad():
            dummy = torch.zeros(1, self._obs_C, self._obs_H, self._obs_W)
            z = self.adaptive_pool(self.conv(dummy))
            conv_out = z.view(1, -1).shape[1]
        self._conv_out = conv_out  # typically 64*7*7 = 3136

        self.fc = nn.Sequential(
            nn.Linear(self._conv_out, self.fc_hidden),
            nn.ReLU(inplace=True),
        )

        # Optional GRU memory
        if self.use_gru:
            self.rnn = nn.GRU(
                input_size=self.fc_hidden,
                hidden_size=self.gru_hidden,
                num_layers=self.gru_layers,
                batch_first=True,
            )
            core_dim = self.gru_hidden
        else:
            self.rnn = None
            core_dim = self.fc_hidden

        # Heads
        self.policy_head = nn.Linear(core_dim, self._num_outputs)
        self.value_head = nn.Linear(core_dim, 1)

    # ---- State handling (new API) ----
    @override(TorchRLModule)
    def get_initial_state(self) -> Dict[str, torch.Tensor]:
        """
        Return per-sequence initial state. RLlib will place this under
        Columns.STATE_IN for the *first* timestep of each sequence and then feed
        back your Columns.STATE_OUT thereafter. Shapes here do NOT include batch.
        """
        device = next(self.parameters()).device
        if self.use_gru:
            # Store per-layer hidden, no batch dim here.
            return {"h": torch.zeros(self.gru_layers, self.gru_hidden, device=device)}
        else:
            return {}  # stateless

    # ---- Public forward methods (RLModule API) ----
    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        logits, _, next_state = self._compute_logits_and_state(batch)
        out = {Columns.ACTION_DIST_INPUTS: logits}
        if next_state:
            out[Columns.STATE_OUT] = next_state
        return out

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Same as inference here; plug in exploration-specific tweaks if desired.
        logits, _, next_state = self._compute_logits_and_state(batch)
        out = {Columns.ACTION_DIST_INPUTS: logits}
        if next_state:
            out[Columns.STATE_OUT] = next_state
        return out

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        logits, embeddings, next_state = self._compute_logits_and_state(batch)
        # Critic predictions required by PPO learners.
        vf = self.compute_values(batch, embeddings=embeddings)  # shape [B*T] or [B,T]
        out = {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: vf,
        }
        if next_state:
            out[Columns.STATE_OUT] = next_state
        return out

    # ---- Value function API ----
    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if embeddings is None:
            embeddings, _ = self._encode(batch)  # [B*T, D]
        v = self.value_head(embeddings).squeeze(-1)  # [B*T]
        return v

    # ---- Helpers ----
    def _compute_logits_and_state(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        embeddings, next_state = self._encode(batch)         # [B*T, D], state dict
        logits = self.policy_head(embeddings)                # [B*T, num_outputs]
        return logits, embeddings, next_state

    def _encode(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """CNN→(optional)GRU encoder that is robust to [B,C,H,W]/[B,T,C,H,W] and channels-last."""
        x = batch[Columns.OBS]
        # Ensure a time dim if absent: [B, ...] -> [B, 1, ...]
        if x.dim() == 4:
            x = x.unsqueeze(1)

        # Reorder HWC to CHW if requested.
        if self.channels_last:
            # [B,T,H,W,C] -> [B,T,C,H,W]
            x = x.permute(0, 1, 4, 2, 3)

        B, T, C, H, W = x.shape
        # Keep original dtype to decide whether to normalize
        orig_dtype = x.dtype
        x = x.reshape(B * T, C, H, W).float()
        if self.frame_norm and orig_dtype in (torch.uint8, torch.int8):
            x = x / 255.0

        z = self.conv(x)               # [B*T, 64, h', w']
        z = self.adaptive_pool(z)      # [B*T, 64, target_spatial, target_spatial]
        z = z.view(B * T, -1)          # [B*T, conv_out]
        z = self.fc(z)                 # [B*T, fc_hidden]

        next_state: Dict[str, torch.Tensor] = {}
        if self.use_gru:
            z_seq = z.view(B, T, -1)   # [B, T, D]

            # Recover state_in (first step only, per sequence) and make it [layers, B, hidden].
            state_in = batch.get(Columns.STATE_IN, None)
            if state_in is not None and "h" in state_in:
                h = state_in["h"]
                # Possible shapes: [layers, hidden] OR [B, hidden] OR [layers, B, hidden]
                if h.dim() == 2 and h.shape == (self.gru_layers, self.gru_hidden):
                    h = h.unsqueeze(1).expand(self.gru_layers, B, self.gru_hidden)
                elif h.dim() == 2 and h.shape == (B, self.gru_hidden):
                    h = h.unsqueeze(0)  # -> [1, B, H] (assume 1 layer)
                elif h.dim() == 3 and h.shape[0] == self.gru_layers:
                    # already [layers, B, hidden]
                    pass
                else:
                    # Fallback: start zeros if shape unexpected
                    device = z_seq.device
                    h = torch.zeros(self.gru_layers, B, self.gru_hidden, device=device)
            else:
                device = z_seq.device
                h = torch.zeros(self.gru_layers, B, self.gru_hidden, device=device)

            # We do not pack/pad here; ConnectorV2 already pads to fixed max_seq_len.
            z_out, h_out = self.rnn(z_seq, h)  # z_out: [B,T,H], h_out: [layers,B,H]
            z = z_out.reshape(B * T, -1)
            # Return next state in a per-timestep-0 style (batch-first, no time dim).
            # For 1 layer, prefer [B, H]; otherwise [layers, B, H] is ok and will be fed back.
            if self.gru_layers == 1:
                next_state["h"] = h_out.squeeze(0).contiguous()  # [B, H]
            else:
                next_state["h"] = h_out.contiguous()             # [layers, B, H]

        return z, next_state
