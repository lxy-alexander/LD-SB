import torch
import torch.nn as nn
import numpy as np


class MultiLayerNet(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 num_layers=1,
                 regime="rich"):
        super().__init__()

        self.regime = regime
        self.num_layers = num_layers

        # First layer (rich init applies)
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.first_act = nn.GELU()

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            self.res_blocks.append(block)

        # Output layer
        self.out = nn.Linear(hidden_dim, num_classes, bias=True)

        self._initialize_weights(input_dim, hidden_dim)

    def _initialize_weights(self, input_dim, hidden_dim):
        # First layer: rich / lazy init
        if self.regime == "rich":
            with torch.no_grad():
                W = torch.randn_like(self.first_layer.weight)
                W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
                self.first_layer.weight.copy_(W)
                self.first_layer.bias.zero_()
        else:
            nn.init.normal_(self.first_layer.weight,
                            mean=0,
                            std=1 / np.sqrt(input_dim))
            nn.init.zeros_(self.first_layer.bias)

        # Residual blocks
        for block in self.res_blocks:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)

        # Output layer
        nn.init.kaiming_normal_(self.out.weight, nonlinearity='linear')
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.first_act(self.first_layer(x))

        for block in self.res_blocks:
            x = x + block(x)  # residual connection

        return self.out(x)

    def get_first_layer_weights(self):
        return self.first_layer.weight.data.clone()
