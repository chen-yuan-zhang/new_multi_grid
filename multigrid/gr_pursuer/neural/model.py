import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def zero_grad(self):
        super().zero_grad()

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """Positional Encoding.

        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.

        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

# Model Definition
class TransformerPredictor(nn.Module):
    def __init__(self, state_dim, goals_dim, mlp_grid_dim, window, num_layers, num_heads, output_dim, enable_positional_encoding=True):
        super().__init__()
        self.grid_mlp = nn.Sequential(
            nn.Linear(mlp_grid_dim[0], mlp_grid_dim[1]),  # First compression layer
            nn.ReLU(),
            nn.Linear(mlp_grid_dim[1], mlp_grid_dim[2]),  # Further reduce dimensions
            nn.ReLU(),
            nn.Linear(mlp_grid_dim[2], mlp_grid_dim[3])  # Match observation length
        )
        self.enable_positional_encoding = enable_positional_encoding
        sequence_dim = state_dim*(window+1)
        input_dim = sequence_dim + mlp_grid_dim[3] + goals_dim
        self.positional_encoding = PositionalEncoding(state_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        grid_encoded = self.grid_mlp(batch['grid'])
        state = batch['state']
        if self.enable_positional_encoding:
            state = self.positional_encoding(batch['state'])
        state = torch.flatten(state, start_dim=1, end_dim=2)
        x = torch.cat([state, grid_encoded, batch['goals']], dim=-1)
        x = self.transformer(x)
        x = self.fc(x)  # Take the last output
        return x
