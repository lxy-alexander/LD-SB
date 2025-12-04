"""
Low-Dimensional Simplicity Bias (LD-SB) Experiments
Based on NeurIPS 2023 paper: "Simplicity Bias in Neural Networks"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Tuple, Dict
import warnings
import os

warnings.filterwarnings('ignore')


class Config:
    """Configuration for LD-SB experiments"""
    data_root = "./imagenette-160"
    output_dir = "./outputs"

    # Model settings
    feature_dim = 2048
    hidden_dim = 100  # 论文用 100
    num_classes = 10

    # Training settings
    batch_size = 128
    num_steps = 20000
    warmup_steps = 500  # 缩短 warmup

    # Rich vs Lazy regime
    regime = "rich"

    # 学习率 - 降低以避免发散
    learning_rate_rich = 1  # 从 0.5 降到 0.1
    learning_rate_lazy = 0.01

    weight_decay = 0.0
    momentum = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42


class OneHiddenLayerNet(nn.Module):
    """One hidden layer neural network for LD-SB experiments."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 regime: str = "rich"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.regime = regime

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to rich or lazy regime"""
        if self.regime == "rich":
            with torch.no_grad():
                # 第一层：每行在单位球面上
                W = torch.randn_like(self.fc1.weight)
                W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
                self.fc1.weight.copy_(W)
                self.fc1.bias.zero_()

                # 第二层：±1/m
                a = torch.randint(0, 2, self.fc2.weight.shape).float() * 2 - 1
                a /= self.hidden_dim
                self.fc2.weight.copy_(a)

        elif self.regime == "lazy":
            nn.init.normal_(self.fc1.weight,
                            mean=0,
                            std=1 / np.sqrt(self.input_dim))
            nn.init.zeros_(self.fc1.bias)
            nn.init.normal_(self.fc2.weight,
                            mean=0,
                            std=1 / np.sqrt(self.hidden_dim))

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def get_first_layer_weights(self):
        return self.fc1.weight.data.clone()


def compute_effective_rank(weight_matrix: torch.Tensor) -> float:
    """Compute effective rank using von Neumann entropy."""
    U, S, V = torch.svd(weight_matrix)
    S_squared = S**2
    S_normalized = S_squared / S_squared.sum()
    S_normalized = S_normalized[S_normalized > 1e-10]
    entropy = -(S_normalized * torch.log(S_normalized)).sum()
    return torch.exp(entropy).item()


def load_imagenette_with_features(
        config: Config,
        split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """Load Imagenette and extract ResNet-50 features."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_path = Path(config.data_root) / split
    dataset = torchvision.datasets.ImageFolder(root=str(data_path),
                                               transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4)

    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(config.device)

    all_features = []
    all_labels = []

    print(f"Extracting features from {split} set...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(config.device)
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


class FeatureDataset(Dataset):

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, config: Config) -> Dict:
    """Train with SGD + momentum, warmup + cosine decay."""
    criterion = nn.CrossEntropyLoss()

    lr = config.learning_rate_rich if config.regime == "rich" else config.learning_rate_lazy
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=config.momentum,
                          weight_decay=config.weight_decay)

    # Warmup + cosine decay
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (config.num_steps -
                                                       config.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'effective_rank': []
    }

    best_val_acc = 0.0

    weight_matrix = model.get_first_layer_weights()
    initial_eff_rank = compute_effective_rank(weight_matrix)
    print(f"\nInitial Effective Rank: {initial_eff_rank:.2f}")
    print("-" * 80)

    step = 0
    train_iter = iter(train_loader)
    steps_per_epoch = len(train_loader)
    eval_interval = steps_per_epoch

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    while step < config.num_steps:
        model.train()

        try:
            features, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            features, labels = next(train_iter)

        features, labels = features.to(config.device), labels.to(config.device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping 防止发散
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        running_total += labels.size(0)
        running_correct += predicted.eq(labels).sum().item()

        step += 1

        if step % eval_interval == 0 or step == config.num_steps:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for val_features, val_labels in val_loader:
                    val_features, val_labels = val_features.to(
                        config.device), val_labels.to(config.device)
                    outputs = model(val_features)
                    loss = criterion(outputs, val_labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += val_labels.size(0)
                    correct += predicted.eq(val_labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100. * correct / total
            train_loss = running_loss / eval_interval
            train_acc = 100. * running_correct / running_total

            weight_matrix = model.get_first_layer_weights()
            eff_rank = compute_effective_rank(weight_matrix)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['effective_rank'].append(eff_rank)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    Path(config.output_dir) /
                    f"best_model_{config.regime}.pth")

            print(
                f"Step [{step}/{config.num_steps}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Acc: {val_acc:.2f}%, Eff.Rank: {eff_rank:.2f}, "
                f"LR: {scheduler.get_last_lr()[0]:.4f}")

            running_loss = 0.0
            running_correct = 0
            running_total = 0

    return history


def find_projection_matrix(model: nn.Module, rank: int) -> torch.Tensor:
    """Find projection matrix P using SVD of first layer weights."""
    weight_matrix = model.get_first_layer_weights()
    U, S, V = torch.svd(weight_matrix)
    V_k = V[:, :rank]
    P = V_k @ V_k.t()
    return P


def evaluate_ld_sb(model: nn.Module, P: torch.Tensor,
                   val_features: torch.Tensor, val_labels: torch.Tensor,
                   config: Config) -> Dict:
    """Evaluate LD-SB metrics."""
    model.eval()

    P = P.to(config.device)
    input_dim = P.shape[0]
    P_perp = torch.eye(input_dim).to(config.device) - P

    n_samples = min(1000, len(val_features))
    indices1 = torch.randint(0, len(val_features), (n_samples, ))
    indices2 = torch.randint(0, len(val_features), (n_samples, ))

    x1 = val_features[indices1].to(config.device)
    x2 = val_features[indices2].to(config.device)

    x_mixed = (P @ x1.t()).t() + (P_perp @ x2.t()).t()

    with torch.no_grad():
        logits_x1 = model(x1)
        logits_x2 = model(x2)
        logits_mixed = model(x_mixed)

        # 使用 softmax 概率而不是原始 logits
        prob_x1 = torch.softmax(logits_x1, dim=1)
        prob_x2 = torch.softmax(logits_x2, dim=1)
        prob_mixed = torch.softmax(logits_mixed, dim=1)

        # P_perp-LC: ||prob(x_mixed) - prob(x1)|| / ||prob(x1)||
        diff_x1 = prob_mixed - prob_x1
        p_perp_lc = (torch.norm(diff_x1, dim=1) /
                     (torch.norm(prob_x1, dim=1) + 1e-8)).mean().item() * 100

        # P-LC
        diff_x2 = prob_mixed - prob_x2
        p_lc = (torch.norm(diff_x2, dim=1) /
                (torch.norm(prob_x2, dim=1) + 1e-8)).mean().item() * 100

        # Prediction change probabilities
        pred_x1 = logits_x1.argmax(dim=1)
        pred_x2 = logits_x2.argmax(dim=1)
        pred_mixed = logits_mixed.argmax(dim=1)

        p_perp_pc = (pred_mixed != pred_x1).float().mean().item() * 100
        p_pc = (pred_mixed != pred_x2).float().mean().item() * 100

    return {
        'P_perp_LC': p_perp_lc,
        'P_LC': p_lc,
        'P_perp_pC': p_perp_pc,
        'P_pC': p_pc,
        'rank_P': torch.linalg.matrix_rank(P).item()
    }


def plot_results(history: Dict, config: Config):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Eval Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Eval Step')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history['effective_rank'])
    axes[2].set_xlabel('Eval Step')
    axes[2].set_ylabel('Effective Rank')
    axes[2].set_title('Effective Rank')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(Path(config.output_dir) / f'training_{config.regime}.png',
                dpi=300)
    plt.close()


def main():
    config = Config()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 80)
    print(
        f"LD-SB Experiment | Regime: {config.regime} | Device: {config.device}"
    )
    print("=" * 80)

    # 1. Load features
    print("\n1. Loading features...")
    train_features, train_labels = load_imagenette_with_features(config,
                                                                 split="train")
    val_features, val_labels = load_imagenette_with_features(config,
                                                             split="val")
    print(f"Train: {train_features.shape}, Val: {val_features.shape}")

    train_loader = DataLoader(FeatureDataset(train_features, train_labels),
                              batch_size=config.batch_size,
                              shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_features, val_labels),
                            batch_size=config.batch_size,
                            shuffle=False)

    # 2. Train
    print(f"\n2. Training ({config.regime} regime)...")
    model = OneHiddenLayerNet(config.feature_dim, config.hidden_dim,
                              config.num_classes,
                              config.regime).to(config.device)
    history = train_model(model, train_loader, val_loader, config)

    # 3. Plot
    print("\n3. Plotting...")
    plot_results(history, config)

    # 4. Find P and evaluate LD-SB
    print("\n4. Evaluating LD-SB...")
    final_eff_rank = history['effective_rank'][-1]
    rank = max(1, int(np.ceil(final_eff_rank)))
    print(f"Using rank k={rank} (effective rank: {final_eff_rank:.2f})")

    P = find_projection_matrix(model, rank)
    metrics = evaluate_ld_sb(model, P, val_features, val_labels, config)

    print("\nLD-SB Metrics:")
    print(f"  rank(P): {metrics['rank_P']}")
    print(f"  P_perp-LC: {metrics['P_perp_LC']:.2f}% (should be low)")
    print(f"  P-LC: {metrics['P_LC']:.2f}% (should be high)")
    print(f"  P_perp-pC: {metrics['P_perp_pC']:.2f}% (should be low)")
    print(f"  P-pC: {metrics['P_pC']:.2f}% (should be high)")

    # 5. Save results
    results = {
        'regime': config.regime,
        'final_val_acc': history['val_acc'][-1],
        'final_effective_rank': final_eff_rank,
        'ldsb_metrics': metrics
    }

    with open(Path(config.output_dir) / f'results_{config.regime}.json',
              'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {config.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
