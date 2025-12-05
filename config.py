# config.py
import torch


class Config:
    data_root = "./waterbird"
    output_dir = "./outputs"

    feature_dim = 2048   # dimensionality of extracted ResNet-50 features
    hidden_dim = 100     # hidden layer width for the MLP classifier
    num_classes = 2      # binary classification setting

    batch_size = 128
    num_steps = 20000
    warmup_steps = 500

    regime = "rich"      # training regime: "rich" or "lazy"

    # Lower the learning rate to avoid collapse when BatchNorm is disabled
    learning_rate_rich = 0.1   # reduced from 1.0 to 0.1 to improve stability
    learning_rate_lazy = 0.01

    momentum = 0.9
    weight_decay = 0.0

    use_batchnorm = False      # disable BatchNorm to match the LD-SB experimental setup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42                  # random seed for reproducibility
