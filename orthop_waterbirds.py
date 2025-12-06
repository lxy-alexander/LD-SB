"""
OrthoP experiment on Waterbirds dataset.
Compares Rich vs Lazy training regimes for 1-hidden layer networks.

This adapts the partner's ldsb/orthop.py methodology for Waterbirds,
establishing the baseline LD-SB behavior before depth experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os

from config import Config
from data import load_cached_or_extract, FeatureDataset
from utils import set_seed


class OneHiddenLayerNet(nn.Module):
    """
    One hidden layer neural network for OrthoP experiments.
    Supports both rich and lazy regime initialization.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, regime="rich"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.regime = regime

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        if self.regime == "rich":
            # Rich: rows sampled uniformly from unit sphere
            with torch.no_grad():
                W = torch.randn_like(self.fc1.weight)
                W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
                self.fc1.weight.copy_(W)
                self.fc1.bias.zero_()
                # Second layer: ±1/m
                a = torch.randint(0, 2, self.fc2.weight.shape).float() * 2 - 1
                a /= self.hidden_dim
                self.fc2.weight.copy_(a)
        else:  # lazy
            nn.init.normal_(self.fc1.weight, mean=0, std=1/np.sqrt(self.input_dim))
            nn.init.zeros_(self.fc1.bias)
            nn.init.normal_(self.fc2.weight, mean=0, std=1/np.sqrt(self.hidden_dim))

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def get_first_layer_weights(self):
        return self.fc1.weight.data.clone()


def compute_effective_rank(W):
    """Compute effective rank using von Neumann entropy."""
    U, S, V = torch.svd(W)
    S2 = S ** 2
    P = S2 / S2.sum()
    P = P[P > 1e-10]
    H = -(P * torch.log(P)).sum()
    return torch.exp(H).item()


def train_model(model, train_loader, val_loader, config, num_epochs=100):
    """Train and track metrics over epochs."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'effective_rank': []
    }

    # Initial rank
    initial_rank = compute_effective_rank(model.get_first_layer_weights())
    print(f"  Initial effective rank: {initial_rank:.2f}")

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total

        # Validate
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(config.device), labels.to(config.device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        # Effective rank
        eff_rank = compute_effective_rank(model.get_first_layer_weights())

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['effective_rank'].append(eff_rank)

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | Val Acc: {val_acc:.2f}% | Eff Rank: {eff_rank:.2f}")

    return history


def find_projection_matrix(model, rank):
    """Find projection matrix P from first layer weights using SVD."""
    W = model.get_first_layer_weights()
    U, S, V = torch.svd(W)
    V_k = V[:, :rank]
    P = V_k @ V_k.t()
    return P


def evaluate_ldsb(model, P, val_features, val_labels, config):
    """Evaluate LD-SB metrics using projection mixing."""
    model.eval()
    P = P.to(config.device)
    P_perp = torch.eye(P.shape[0]).to(config.device) - P

    n_samples = min(1000, len(val_features))
    idx1 = torch.randint(0, len(val_features), (n_samples,))
    idx2 = torch.randint(0, len(val_features), (n_samples,))

    x1 = val_features[idx1].to(config.device)
    x2 = val_features[idx2].to(config.device)

    # Mixed input: P*x1 + P_perp*x2
    x_mixed = (P @ x1.t()).t() + (P_perp @ x2.t()).t()

    with torch.no_grad():
        logits_x1 = model(x1)
        logits_x2 = model(x2)
        logits_mixed = model(x_mixed)

        # P_perp-LC
        diff_x1 = logits_mixed - logits_x1
        p_perp_lc = (torch.norm(diff_x1, dim=1) / (torch.norm(logits_x1, dim=1) + 1e-8)).mean().item() * 100

        # P-LC
        diff_x2 = logits_mixed - logits_x2
        p_lc = (torch.norm(diff_x2, dim=1) / (torch.norm(logits_x2, dim=1) + 1e-8)).mean().item() * 100

        # P_perp-pC
        pred_x1 = logits_x1.argmax(dim=1)
        pred_mixed = logits_mixed.argmax(dim=1)
        p_perp_pc = (pred_mixed != pred_x1).float().mean().item() * 100

        # P-pC
        pred_x2 = logits_x2.argmax(dim=1)
        p_pc = (pred_mixed != pred_x2).float().mean().item() * 100

    return {
        'P_perp_LC': p_perp_lc,
        'P_LC': p_lc,
        'P_perp_pC': p_perp_pc,
        'P_pC': p_pc,
        'rank_P': int(torch.linalg.matrix_rank(P).item())
    }


def plot_rank_comparison(history_rich, history_lazy, save_path):
    """Plot effective rank evolution for both regimes."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history_rich['effective_rank']) + 1)

    plt.plot(epochs, history_rich['effective_rank'], 
             label='Rich Regime', linewidth=2, color='#2E86AB')
    plt.plot(epochs, history_lazy['effective_rank'], 
             label='Lazy Regime', linewidth=2, color='#E94F37')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Effective Rank', fontsize=12)
    plt.title('OrthoP: Effective Rank Evolution on Waterbirds', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    config = Config()
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("OrthoP Experiment: Rich vs Lazy on Waterbirds")
    print("=" * 60)

    # Load features
    train_x, train_y = load_cached_or_extract(config, "train")
    val_x, val_y = load_cached_or_extract(config, "val")

    train_loader = DataLoader(FeatureDataset(train_x, train_y),
                              batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_x, val_y),
                            batch_size=config.batch_size, shuffle=False)

    print(f"Train: {len(train_x)} samples, Val: {len(val_x)} samples")

    results = {}

    for regime in ["rich", "lazy"]:
        print(f"\n{'='*60}")
        print(f"Training {regime.upper()} regime (1 hidden layer)")
        print("=" * 60)

        model = OneHiddenLayerNet(
            input_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            regime=regime
        ).to(config.device)

        history = train_model(model, train_loader, val_loader, config, num_epochs=100)

        # Find projection matrix
        final_rank = history['effective_rank'][-1]
        k = max(1, int(np.ceil(final_rank)))
        P = find_projection_matrix(model, k)

        # Evaluate LD-SB
        ldsb_metrics = evaluate_ldsb(model, P, val_x, val_y, config)

        results[regime] = {
            'history': history,
            'ldsb_metrics': ldsb_metrics,
            'final_val_acc': history['val_acc'][-1],
            'final_effective_rank': final_rank
        }

        print(f"\nFinal Results ({regime}):")
        print(f"  Val Acc: {history['val_acc'][-1]:.2f}%")
        print(f"  Eff Rank: {final_rank:.2f}")
        print(f"  P⊥-pC: {ldsb_metrics['P_perp_pC']:.1f}%")
        print(f"  P-pC: {ldsb_metrics['P_pC']:.1f}%")

    # Plot comparison
    plot_rank_comparison(
        results['rich']['history'],
        results['lazy']['history'],
        "results/orthop_waterbirds_rank.png"
    )

    # Print comparison table
    print("\n" + "=" * 70)
    print("OrthoP COMPARISON TABLE (Waterbirds, 1 Hidden Layer)")
    print("=" * 70)
    print(f"{'Regime':<10} {'Val Acc':<12} {'Eff Rank':<12} {'P⊥-pC':<10} {'P-pC':<10}")
    print("-" * 70)
    for regime in ["rich", "lazy"]:
        r = results[regime]
        m = r['ldsb_metrics']
        print(f"{regime.capitalize():<10} {r['final_val_acc']:.2f}%{'':<5} "
              f"{r['final_effective_rank']:.2f}{'':<7} "
              f"{m['P_perp_pC']:.1f}%{'':<5} {m['P_pC']:.1f}%")
    print("=" * 70)

    # Save results
    save_results = {
        'rich': {
            'final_val_acc': results['rich']['final_val_acc'],
            'final_effective_rank': results['rich']['final_effective_rank'],
            'ldsb_metrics': results['rich']['ldsb_metrics'],
            'history': {
                'effective_rank': results['rich']['history']['effective_rank'],
                'val_acc': results['rich']['history']['val_acc']
            }
        },
        'lazy': {
            'final_val_acc': results['lazy']['final_val_acc'],
            'final_effective_rank': results['lazy']['final_effective_rank'],
            'ldsb_metrics': results['lazy']['ldsb_metrics'],
            'history': {
                'effective_rank': results['lazy']['history']['effective_rank'],
                'val_acc': results['lazy']['history']['val_acc']
            }
        }
    }

    with open("outputs/orthop_waterbirds.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to outputs/orthop_waterbirds.json")


if __name__ == "__main__":
    main()

