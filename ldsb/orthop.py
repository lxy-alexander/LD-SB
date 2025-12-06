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
from typing import Tuple, Dict, Optional
import warnings
import os

warnings.filterwarnings('ignore')


class Config:
    """Configuration for LD-SB experiments"""
    # Paths - CHANGE THESE TO YOUR PATHS
    data_root = "./imagenette-320"  # Path to your Imagenette dataset
    output_dir = "./outputs"  # Where to save results

    # Model settings
    feature_dim = 2048  # ResNet-50 output dimension
    hidden_dim = 512
    num_classes = 10

    # Training settings
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-4

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seed
    seed = 42


class OneHiddenLayerNet(nn.Module):
    """
    One hidden layer neural network for LD-SB experiments.
    Supports both rich and lazy regime initialization.
    """

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
            # Rich Initialization: Each row sampled uniformly from the unit sphere
            with torch.no_grad():
                W = torch.randn_like(self.fc1.weight)  # shape: (m, d)
                W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
                self.fc1.weight.copy_(W)
                self.fc1.bias.zero_()

                # Second layer: ±1 uniformly, scaled by 1/m
                a = torch.randint(0, 2, self.fc2.weight.shape).float() * 2 - 1
                a /= self.hidden_dim
                self.fc2.weight.copy_(a)

        elif self.regime == "lazy":
            # Lazy regime: Kaiming-like initialization
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
        """Get first layer weight matrix for analysis"""
        return self.fc1.weight.data.clone()


def compute_effective_rank(weight_matrix: torch.Tensor) -> float:
    """
    Compute effective rank of a matrix as defined in Definition 4.3.
    Effective rank = exp(-sum(p_i * log(p_i)))
    where p_i are normalized singular values.
    """
    # Compute singular values
    U, S, V = torch.svd(weight_matrix)

    # Normalize singular values
    S_squared = S**2
    S_normalized = S_squared / S_squared.sum()

    # Compute von Neumann entropy (handle zeros)
    S_normalized = S_normalized[S_normalized > 1e-10]
    entropy = -(S_normalized * torch.log(S_normalized)).sum()
    effective_rank = torch.exp(entropy).item()

    return effective_rank


def load_imagenette_with_features(
        config: Config,
        split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load Imagenette dataset and extract ResNet-50 features.
    Returns: (features, labels)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    data_path = Path(config.data_root) / split
    dataset = torchvision.datasets.ImageFolder(root=str(data_path),
                                               transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4)

    # Load pretrained ResNet-50
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Remove final classification layer to get features
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(config.device)

    # Extract features
    all_features = []
    all_labels = []

    print(f"Extracting features from {split} set...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(config.device)
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)  # Flatten
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


class FeatureDataset(Dataset):
    """Dataset wrapper for extracted features"""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, config: Config) -> Dict:
    """
    Train the one-hidden layer network and track metrics.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=config.num_epochs)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'effective_rank': []
    }

    best_val_acc = 0.0

    # Print initial effective rank before training
    weight_matrix = model.get_first_layer_weights()
    initial_eff_rank = compute_effective_rank(weight_matrix)
    print(
        f"\nInitial Effective Rank (before training): {initial_eff_rank:.2f}")
    print("-" * 80)

    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(config.device), labels.to(
                config.device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(config.device), labels.to(
                    config.device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        # Compute effective rank
        weight_matrix = model.get_first_layer_weights()
        eff_rank = compute_effective_rank(weight_matrix)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['effective_rank'].append(eff_rank)

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                Path(config.output_dir) / f"best_model_{model.regime}.pth")

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{config.num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"Eff. Rank: {eff_rank:.2f}")

    return history


def find_projection_matrix_rich(model: nn.Module, rank: int) -> torch.Tensor:
    """
    Find projection matrix P for rich regime using SVD of first layer weights.
    Returns P (projection onto top-k subspace) of shape (input_dim, input_dim).
    """
    weight_matrix = model.get_first_layer_weights(
    )  # shape: (hidden_dim, input_dim)

    # Compute SVD
    U, S, V = torch.svd(weight_matrix)

    # Take top-k right singular vectors
    V_k = V[:, :rank]  # shape: (input_dim, rank)

    # Construct projection matrix P = V_k @ V_k^T
    P = V_k @ V_k.t()

    return P


def find_projection_matrix_lazy(model: nn.Module, train_features: torch.Tensor,
                                train_labels: torch.Tensor, rank: int,
                                config: Config) -> torch.Tensor:
    """
    Find projection matrix P for lazy regime.
    For lazy regime, we use PCA on the feature space as the model stays close to initialization.
    This captures the dominant directions in the input space.
    """
    device = config.device
    X = train_features.to(device)

    print(
        f"Computing PCA-based projection for Lazy regime with rank={rank}...")

    # Center the data
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    # Compute SVD
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    # Take top-k principal components
    V_k = Vh[:rank].T  # shape: (d, rank)

    # Construct projection matrix P = V_k @ V_k^T
    P = V_k @ V_k.T

    print(f"Lazy projection matrix computed with rank={rank}")

    return P.cpu()


def evaluate_ld_sb(model: nn.Module, P: torch.Tensor,
                   val_features: torch.Tensor, val_labels: torch.Tensor,
                   config: Config) -> Dict:
    """
    Evaluate LD-SB metrics on validation set.
    Metrics:
    - P_perp-LC: relative logit change w.r.t. x1
    - P-LC: relative logit change w.r.t. x2
    - P_perp-pC: prediction change probability w.r.t. x1
    - P-pC: prediction change probability w.r.t. x2
    """
    model.eval()
    P = P.to(config.device)

    input_dim = P.shape[0]
    P_perp = torch.eye(input_dim).to(config.device) - P

    n_samples = min(1000, len(val_features))
    indices1 = torch.randint(0, len(val_features), (n_samples, ))
    indices2 = torch.randint(0, len(val_features), (n_samples, ))

    x1 = val_features[indices1].to(config.device)
    x2 = val_features[indices2].to(config.device)
    y1 = val_labels[indices1].to(config.device)

    # Create mixed input: P*x1 + P_perp*x2
    x_mixed = (P @ x1.t()).t() + (P_perp @ x2.t()).t()

    with torch.no_grad():
        # Get model outputs
        logits_x1 = model(x1)
        logits_x2 = model(x2)
        logits_mixed = model(x_mixed)

        # Compute metrics
        # P_perp-LC: ||f(x_mixed) - f(x1)|| / ||f(x1)||
        diff_x1 = logits_mixed - logits_x1
        norm_diff_x1 = torch.norm(diff_x1, dim=1)
        norm_logits_x1 = torch.norm(logits_x1, dim=1)
        p_perp_lc = (norm_diff_x1 /
                     (norm_logits_x1 + 1e-8)).mean().item() * 100

        # P-LC: ||f(x_mixed) - f(x2)|| / ||f(x2)||
        diff_x2 = logits_mixed - logits_x2
        norm_diff_x2 = torch.norm(diff_x2, dim=1)
        norm_logits_x2 = torch.norm(logits_x2, dim=1)
        p_lc = (norm_diff_x2 / (norm_logits_x2 + 1e-8)).mean().item() * 100

        # P_perp-pC: P[pred(f(x_mixed)) != pred(f(x1))]
        pred_x1 = logits_x1.argmax(dim=1)
        pred_mixed = logits_mixed.argmax(dim=1)
        p_perp_pc = (pred_mixed != pred_x1).float().mean().item() * 100

        # P-pC: P[pred(f(x_mixed)) != pred(f(x2))]
        pred_x2 = logits_x2.argmax(dim=1)
        p_pc = (pred_mixed != pred_x2).float().mean().item() * 100

    results = {
        'P_perp_LC': p_perp_lc,
        'P_LC': p_lc,
        'P_perp_pC': p_perp_pc,
        'P_pC': p_pc,
        'rank_P': torch.linalg.matrix_rank(P).item()
    }

    return results


def plot_effective_rank_comparison(history_rich: Dict, history_lazy: Dict,
                                   config: Config):
    """Plot effective rank evolution for both rich and lazy regimes"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(history_rich['effective_rank']) + 1)

    plt.plot(epochs,
             history_rich['effective_rank'],
             label='Rich Regime',
             linewidth=2,
             color='blue')
    plt.plot(epochs,
             history_lazy['effective_rank'],
             label='Lazy Regime',
             linewidth=2,
             color='red')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Effective Rank', fontsize=12)
    plt.title('Effective Rank Evolution: Rich vs Lazy Regime',
              fontsize=14,
              fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(Path(config.output_dir) / 'effective_rank_comparison.png',
                dpi=300)
    plt.close()
    print(
        f"\nEffective rank comparison plot saved to {config.output_dir}/effective_rank_comparison.png"
    )


def print_comparison_table(metrics_rich: Dict, metrics_lazy: Dict):
    """Print comparison table of LD-SB metrics"""
    print("\n" + "=" * 80)
    print("LD-SB METRICS COMPARISON TABLE")
    print("=" * 80)
    print(
        f"{'Dataset':<15} {'rank(P)':<12} {'P⊥-LC (↓)':<15} {'P-LC (↑)':<15} {'P⊥-pC (↓)':<15} {'P-pC (↑)':<15}"
    )
    print("-" * 80)

    # Rich regime
    print(f"{'Rich':<15} "
          f"{metrics_rich['rank_P']:<12} "
          f"{metrics_rich['P_perp_LC']:<15.2f} "
          f"{metrics_rich['P_LC']:<15.2f} "
          f"{metrics_rich['P_perp_pC']:<15.2f} "
          f"{metrics_rich['P_pC']:<15.2f}")

    # Lazy regime
    print(f"{'Lazy':<15} "
          f"{metrics_lazy['rank_P']:<12} "
          f"{metrics_lazy['P_perp_LC']:<15.2f} "
          f"{metrics_lazy['P_LC']:<15.2f} "
          f"{metrics_lazy['P_perp_pC']:<15.2f} "
          f"{metrics_lazy['P_pC']:<15.2f}")

    print("=" * 80)
    print("\nMetric Interpretation:")
    print(
        "  - P⊥-LC (↓): Should be LOW (small change when perturbing in P⊥ direction)"
    )
    print(
        "  - P-LC (↑): Should be HIGH (large change when perturbing in P direction)"
    )
    print(
        "  - P⊥-pC (↓): Should be LOW (prediction stable to P⊥ perturbations)")
    print(
        "  - P-pC (↑): Should be HIGH (prediction changes with P perturbations)"
    )
    print("=" * 80 + "\n")


class MNISTCIFAR(Dataset):
    """
    MNIST-CIFAR collage dataset as used in Shah et al. (2020).
    Output shape: (3, 32, 64)
    """

    def __init__(self, split="train"):
        super().__init__()

        # Transforms
        self.transform_mnist = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        self.transform_cifar = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load MNIST
        self.mnist = torchvision.datasets.MNIST(root="./mnist",
                                                train=(split == "train"),
                                                download=True)

        # Load CIFAR-10
        self.cifar = torchvision.datasets.CIFAR10(root="./cifar10",
                                                  train=(split == "train"),
                                                  download=True)

        assert len(self.mnist) == len(
            self.cifar), "Datasets must have same length"
        self.length = len(self.mnist)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load MNIST
        mnist_img, mnist_label = self.mnist[idx]
        mnist_img = self.transform_mnist(mnist_img)  # shape: (3, 32, 32)

        # Load CIFAR
        cifar_img, cifar_label = self.cifar[idx]
        cifar_img = self.transform_cifar(cifar_img)  # shape: (3, 32, 32)

        # Concatenate along width → (3, 32, 64)
        concat_img = torch.cat([mnist_img, cifar_img], dim=2)

        # By Shah et al., either label can be used; usually CIFAR label is used.
        label = cifar_label

        return concat_img, label


def load_mnist_cifar_with_features(config: Config, split: str = "train"):
    """
    MNIST-CIFAR (Shah et al., 2020) feature extraction.
    Returns: (features, labels)
    """
    dataset = MNISTCIFAR(split=split)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4)

    # Resize and normalize for ResNet-50
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ResNet feature extractor
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.to(config.device)
    feature_extractor.eval()

    all_features, all_labels = [], []

    print(f"Extracting features for MNIST-CIFAR ({split})...")
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            # Move to device
            img = img.to(config.device)
            label = label.to(config.device)

            # Resize + normalize
            img = nn.functional.interpolate(img,
                                            size=(224, 224),
                                            mode="bilinear")
            img = preprocess(img)

            # Extract features
            feat = feature_extractor(img)
            feat = feat.view(feat.size(0), -1)

            all_features.append(feat.cpu())
            all_labels.append(label.cpu())

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    return features, labels


def main():
    """Main execution function"""
    config = Config()

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(config.output_dir)}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        print("Using current directory")
        config.output_dir = "."

    print("=" * 80)
    print("Low-Dimensional Simplicity Bias (LD-SB) Experiments")
    print("Comparing Rich vs Lazy Regimes")
    print(f"Device: {config.device}")
    print("=" * 80)

    # Load and extract features
    print(
        "\n1. Loading Imagenette dataset and extracting ResNet-50 features...")
    # train_features, train_labels = load_imagenette_with_features(config, split="train")
    # val_features, val_labels = load_imagenette_with_features(config, split="val")
    train_features, train_labels = load_mnist_cifar_with_features(
        config, split="train")
    val_features, val_labels = load_mnist_cifar_with_features(config,
                                                              split="test")

    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")

    # Create dataloaders
    train_dataset = FeatureDataset(train_features, train_labels)
    val_dataset = FeatureDataset(val_features, val_labels)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False)

    # Dictionary to store results
    results_dict = {}

    # Train both regimes
    for regime in ["rich", "lazy"]:
        print(f"\n{'='*80}")
        print(f"TRAINING {regime.upper()} REGIME")
        print(f"{'='*80}")

        # Create model
        model = OneHiddenLayerNet(config.feature_dim, config.hidden_dim,
                                  config.num_classes, regime).to(config.device)

        # Train model
        history = train_model(model, train_loader, val_loader, config)

        # Find projection matrix P
        print(f"\nFinding projection matrix P for {regime} regime...")
        if regime == "rich":
            final_eff_rank = history['effective_rank'][-1]
            rank = max(1, int(np.ceil(final_eff_rank)))
            print(
                f"Using rank k={rank} (based on effective rank: {final_eff_rank:.2f})"
            )
            P = find_projection_matrix_rich(model, rank)
        else:  # lazy
            rank = max(1,
                       int(config.feature_dim * 0.01))  # Use 1% of dimensions
            print(f"Using rank k={rank}")
            P = find_projection_matrix_lazy(model, train_features,
                                            train_labels, rank, config)

        # Evaluate LD-SB metrics
        print(f"\nEvaluating LD-SB metrics for {regime} regime...")
        ldsb_metrics = evaluate_ld_sb(model, P, val_features, val_labels,
                                      config)

        # Store results
        results_dict[regime] = {
            'history': history,
            'ldsb_metrics': ldsb_metrics,
            'model': model,
            'P': P
        }

    # Plot effective rank comparison
    print("\n2. Generating effective rank comparison plot...")
    plot_effective_rank_comparison(results_dict['rich']['history'],
                                   results_dict['lazy']['history'], config)

    # Print comparison table
    print_comparison_table(results_dict['rich']['ldsb_metrics'],
                           results_dict['lazy']['ldsb_metrics'])

    # Save results to JSON
    results_summary = {
        'rich': {
            'final_train_acc':
            results_dict['rich']['history']['train_acc'][-1],
            'final_val_acc':
            results_dict['rich']['history']['val_acc'][-1],
            'final_effective_rank':
            results_dict['rich']['history']['effective_rank'][-1],
            'ldsb_metrics':
            results_dict['rich']['ldsb_metrics']
        },
        'lazy': {
            'final_train_acc':
            results_dict['lazy']['history']['train_acc'][-1],
            'final_val_acc':
            results_dict['lazy']['history']['val_acc'][-1],
            'final_effective_rank':
            results_dict['lazy']['history']['effective_rank'][-1],
            'ldsb_metrics':
            results_dict['lazy']['ldsb_metrics']
        }
    }

    with open(Path(config.output_dir) / 'results_comparison.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {config.output_dir}/results_comparison.json")
    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
