import torch
import torch.nn as nn
import numpy as np


class MultiLayerNet(nn.Module):
    # MLP architecture used in the LD-SB experiments.
    # ReLU activations, no normalization layers, and no residual connections.

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 num_layers=1,
                 regime="rich"):
        super().__init__()

        self.regime = regime
        self.num_layers = num_layers

        layers = []

        # First hidden layer: input → hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Additional hidden layers (fully connected, no residuals)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        # Output classifier layer
        self.out = nn.Linear(hidden_dim, num_classes)

        # Initialize weights following rich/lazy regime rules
        self._initialize_weights(input_dim)

        # Report initial effective rank for monitoring
        initial_rank = self._compute_effective_rank()
        print(f"Model: {num_layers} layer(s), regime={regime}")
        print(f"  First layer shape: ({hidden_dim}, {input_dim})")
        print(f"  Initial effective rank: {initial_rank:.2f}")

    def _compute_effective_rank(self):
        # Computes the effective rank of the first-layer weight matrix,
        # defined as exp(Shannon entropy of normalized singular values).
        W = self.get_first_layer_weights()
        U, S, V = torch.svd(W)

        S = S[S > 1e-10]  # Filter numerical noise
        S2 = S**2
        P = S2 / S2.sum()

        H = -(P * torch.log(P)).sum()  # von Neumann / Shannon entropy
        return torch.exp(H).item()

    def _initialize_weights(self, input_dim):
        # Apply initialization strategies corresponding to rich or lazy regimes.

        first_linear = True

        for layer in self.features:
            if isinstance(layer, nn.Linear):

                # Initialization for the first layer
                if first_linear:
                    if self.regime == "rich":
                        # Rich regime: initialize weights with normalized random vectors
                        with torch.no_grad():
                            W = torch.randn_like(layer.weight)
                            W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
                            layer.weight.copy_(W)
                            layer.bias.zero_()

                    else:
                        # Lazy regime: NTK-style variance scaling
                        nn.init.normal_(layer.weight,
                                        mean=0,
                                        std=1 / np.sqrt(input_dim))
                        nn.init.zeros_(layer.bias)

                    first_linear = False

                else:
                    # Hidden layers: standard Kaiming initialization
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)

        # Output layer initialization
        nn.init.kaiming_normal_(self.out.weight, nonlinearity='linear')
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        # Standard forward pass: feature extractor → classifier
        x = self.features(x)
        return self.out(x)

    def get_first_layer_weights(self):
        # Returns the weight matrix of the first Linear layer in the MLP
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                return layer.weight.data.clone()
