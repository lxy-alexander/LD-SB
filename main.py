import argparse
import os
import torch
from config import Config
from model import MultiLayerNet
from data import create_feature_loaders
from train import train_model
from ldsb_eval import find_projection_matrix, evaluate_ldsb


def train():

    # Load dataset features
    train_loader, val_loader, val_x, val_y = create_feature_loaders(config)

    # Build model
    model = MultiLayerNet(
        input_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        num_layers=NUM_LAYERS,
        regime=config.regime
    ).to(config.device)

    # Train
    history = train_model(model, train_loader, val_loader, config)

    # Rank & LD-SB evaluation
    final_rank = history["effective_rank"][-1]
    k = max(1, int(final_rank))

    P = find_projection_matrix(model, k)
    metrics = evaluate_ldsb(model, P, val_x, val_y, config)

    print("LD-SB metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--layers", type=int, default=1,
                        help="Number of hidden layers in the MLP")

    parser.add_argument("--dataset", type=str, default="imagenette160",
                        help="Dataset name (e.g. imagenette160, cifar, custom)")

    parser.add_argument("--epoch", type=int, default=10,
                        help="Number of training epochs")

    args = parser.parse_args()

    NUM_LAYERS = args.layers
    DATASET_NAME = args.dataset
    EPOCHS = args.epoch

    # load default config
    config = Config()

    # override config fields by user arguments
    config.num_epochs = EPOCHS
    config.dataset = DATASET_NAME

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Running LD-SB experiment:")
    print(f"  layers   = {NUM_LAYERS}")
    print(f"  dataset  = {DATASET_NAME}")
    print(f"  epochs   = {EPOCHS}")
    
    train()
