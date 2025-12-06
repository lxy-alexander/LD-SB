"""
Main entry point for LD-SB experiments on Waterbirds dataset.

This script trains multi-layer MLPs with configurable depth and training regime
(rich or lazy), then evaluates low-dimensional simplicity bias (LD-SB) metrics.

Usage:
    python main.py --layers 5 --regime rich --lr 0.2
    python main.py --layers 10 --regime lazy

Results are saved to outputs/results_{regime}_layer{layers}.json
"""

import argparse
import json
import os
import torch
from config import Config
from model import MultiLayerNet
from data import create_feature_loaders
from train import train_model
from ldsb_eval import find_projection_matrix, evaluate_ldsb
from utils import set_seed


def train(config, num_layers, lr):
    """
    Train a multi-layer MLP and evaluate LD-SB metrics.

    Args:
        config: Configuration object with hyperparameters
        num_layers: Number of hidden layers in the MLP
        lr: Learning rate for training

    Returns:
        None. Results are saved to JSON file in config.output_dir.
    """

    train_loader, val_loader, val_x, val_y = create_feature_loaders(config)

    model = MultiLayerNet(input_dim=config.feature_dim,
                          hidden_dim=config.hidden_dim,
                          num_classes=config.num_classes,
                          num_layers=num_layers,
                          regime=config.regime).to(config.device)

    history = train_model(model, train_loader, val_loader, config)

    final_rank = history["effective_rank"][-1]
    k = max(1, int(final_rank))

    P = find_projection_matrix(model, k)
    metrics = evaluate_ldsb(model, P, val_x, val_y, config)

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"LD-SB metrics: {metrics}")

    # Save results to JSON
    results = {
        "config": {
            "num_layers": num_layers,
            "regime": config.regime,
            "learning_rate": lr,
            "num_steps": config.num_steps,
            "hidden_dim": config.hidden_dim,
            "seed": config.seed,
        },
        "history": history,
        "ldsb_metrics": metrics,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "final_effective_rank": final_rank,
    }

    output_path = os.path.join(config.output_dir, f"results_{config.regime}_layer{num_layers}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--regime", type=str, default=None, choices=["rich", "lazy"],
                        help="Training regime: 'rich' or 'lazy'")
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    if args.regime is not None:
        config.regime = args.regime

    if args.lr is not None:
        config.learning_rate_rich = args.lr
        config.learning_rate_lazy = args.lr

    if args.steps is not None:
        config.num_steps = args.steps

    os.makedirs(config.output_dir, exist_ok=True)

    lr = config.learning_rate_rich if config.regime == "rich" else config.learning_rate_lazy

    print(f"=" * 60)
    print(f"  layers = {args.layers}")
    print(f"  regime = {config.regime}")
    print(f"  lr     = {lr}")
    print(f"  steps  = {config.num_steps}")
    print(f"=" * 60)

    train(config, args.layers, lr)
