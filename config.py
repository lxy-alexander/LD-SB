# config.py
import torch


class Config:
    """Configuration for LD-SB experiments"""
    data_root = "./imagenette-160"
    output_dir = "./outputs"

    # Model settings
    feature_dim = 2048
    hidden_dim = 100
    num_classes = 10

    # Training settings
    batch_size = 128
    num_steps = 10000
    warmup_steps = 500

    # Regime
    regime = "rich"  # or "lazy"

    # Learning rates
    learning_rate_rich = 1.0
    learning_rate_lazy = 0.01

    momentum = 0.9
    weight_decay = 0.0

    # Device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
