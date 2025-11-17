"""
Low-Dimensional Simplicity Bias (LD-SB) Experiments
Based on NeurIPS 2023 paper: "Simplicity Bias in Neural Networks"

This implementation demonstrates LD-SB on the Imagenette dataset.
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
from typing import Tuple, Dict, Optional
import warnings
import os
warnings.filterwarnings('ignore')


class Config:
    """Configuration for LD-SB experiments"""
    # Paths - CHANGE THESE TO YOUR PATHS
    data_root = "./imagenette-160"  # Path to your Imagenette dataset
    output_dir = "./outputs"     # Where to save results
    
    # Model settings
    feature_dim = 2048  # ResNet-50 output dimension
    hidden_dim = 512
    num_classes = 10
    
    # Training settings
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-4
    
    # Rich vs Lazy regime
    regime = "rich"  # "rich" or "lazy"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random seed
    seed = 42


class OneHiddenLayerNet(nn.Module):
    """
    One hidden layer neural network for LD-SB experiments.
    Supports both rich and lazy regime initialization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, regime: str = "rich"):
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
            # --------- Strict Rich Initialization (Paper version) ---------
            # Each row sampled uniformly from the unit sphere S^d
            with torch.no_grad():
                W = torch.randn_like(self.fc1.weight)  # shape: (m, d)
                W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
                self.fc1.weight.copy_(W)

                # biases from unit sphere in R? but theory allows arbitrary
                self.fc1.bias.zero_()

                # Second layer: ±1 uniformly, scaled by 1/m as in paper
                a = torch.randint(0, 2, self.fc2.weight.shape).float() * 2 - 1
                a /= self.hidden_dim  # scale 1/m
                self.fc2.weight.copy_(a)

            
            # Second layer: uniform {-1, +1} scaled by 1/m
            with torch.no_grad():
                self.fc2.weight.data = torch.randint(0, 2, self.fc2.weight.shape).float() * 2 - 1
                self.fc2.weight.data /= self.hidden_dim
        
        elif self.regime == "lazy":
            # Lazy regime: Kaiming-like initialization
            nn.init.normal_(self.fc1.weight, mean=0, std=1/np.sqrt(self.input_dim))
            nn.init.zeros_(self.fc1.bias)
            nn.init.normal_(self.fc2.weight, mean=0, std=1/np.sqrt(self.hidden_dim))
    
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
    Effective rank = exp(-sum(p_i * log(p_i))) where p_i are normalized singular values.
    """
    # Compute singular values
    U, S, V = torch.svd(weight_matrix)
    
    # Normalize singular values
    S_squared = S ** 2
    S_normalized = S_squared / S_squared.sum()
    
    # Compute von Neumann entropy (handle zeros)
    S_normalized = S_normalized[S_normalized > 1e-10]
    entropy = -(S_normalized * torch.log(S_normalized)).sum()
    
    effective_rank = torch.exp(entropy).item()
    return effective_rank


def load_imagenette_with_features(config: Config, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load Imagenette dataset and extract ResNet-50 features.
    Returns: (features, labels)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    data_path = Path(config.data_root) / split
    dataset = torchvision.datasets.ImageFolder(root=str(data_path), transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
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


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                config: Config) -> Dict:
    """
    Train the one-hidden layer network and track metrics.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
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
    print(f"\nInitial Effective Rank (before training): {initial_eff_rank:.2f}")
    print("-" * 80)
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(config.device), labels.to(config.device)
            
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
                features, labels = features.to(config.device), labels.to(config.device)
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
            torch.save(model.state_dict(), Path(config.output_dir) / f"best_model_{config.regime}.pth")
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"Eff. Rank: {eff_rank:.2f}")
    
    return history


def find_projection_matrix_rich(model: nn.Module, rank: int) -> torch.Tensor:
    """
    Find projection matrix P for rich regime using SVD of first layer weights.
    Returns P (projection onto top-k subspace) of shape (input_dim, input_dim).
    """
    weight_matrix = model.get_first_layer_weights()  # shape: (hidden_dim, input_dim)
    
    # Compute SVD
    U, S, V = torch.svd(weight_matrix)
    
    # Take top-k right singular vectors
    V_k = V[:, :rank]  # shape: (input_dim, rank)
    
    # Construct projection matrix P = V_k @ V_k^T
    P = V_k @ V_k.t()
    
    return P

def find_projection_matrix_lazy(model: nn.Module, 
                                train_features: torch.Tensor, 
                                train_labels: torch.Tensor,
                                rank: int,
                                config: Config,
                                lambda_reg: float = 1.0,
                                lr: float = 0.01,
                                iters: int = 500):

    """
    Minimal-corrected version that matches the paper more closely:
    - Optimize V (input_dim x rank) instead of P
    - Ensure projection matrix P = V V^T
    - After each gradient step, orthogonalize V by QR
    - Use full batch loss (paper requirement)
    """

    device = config.device
    X = train_features.to(device)
    Y = train_labels.to(device)
    n, d = X.shape

    # Initialize V using PCA (good starting point)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V_param = Vh[:rank].T.contiguous()  # shape: (d, rank)
    V_param = nn.Parameter(V_param.to(device))

    optimizer = optim.Adam([V_param], lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.eval()

    print(f"Optimizing Lazy P with rank={rank}, full batch...")

    for it in range(iters):
        optimizer.zero_grad()

        # Build projection P = V V^T
        P = V_param @ V_param.T  # (d,d)
        P_perp = torch.eye(d, device=device) - P

        # Project full batch
        Xp = X @ P
        Xp_perp = X @ P_perp

        with torch.no_grad():
            logits_P = model(Xp)
            logits_P_perp = model(Xp_perp)

        # Loss 1: correct prediction on P*x
        loss1 = criterion(logits_P, Y)

        # Loss 2: uniform logits on P_perp * x
        uniform = torch.ones_like(logits_P_perp) / config.num_classes
        loss2 = -torch.mean(torch.sum(uniform * torch.log_softmax(logits_P_perp, dim=1), dim=1))

        loss = loss1 + lambda_reg * loss2
        loss.backward()
        optimizer.step()

        # ---- Important: retraction to keep V orthonormal ----
        with torch.no_grad():
            Q, _ = torch.linalg.qr(V_param)
            V_param.copy_(Q[:, :rank])

        if (it + 1) % 100 == 0:
            print(f"Iter {it+1}/{iters} Loss={loss.item():.4f}")

    # Return final projection matrix P
    P_final = (V_param @ V_param.T).detach().cpu()
    return P_final

def evaluate_ld_sb(model: nn.Module, P: torch.Tensor, val_features: torch.Tensor, 
                   val_labels: torch.Tensor, config: Config) -> Dict:
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
    indices1 = torch.randint(0, len(val_features), (n_samples,))
    indices2 = torch.randint(0, len(val_features), (n_samples,))
    
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
        p_perp_lc = (norm_diff_x1 / (norm_logits_x1 + 1e-8)).mean().item() * 100
        
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


def train_orthop_model(P: torch.Tensor, train_features: torch.Tensor, train_labels: torch.Tensor,
                       val_features: torch.Tensor, val_labels: torch.Tensor, config: Config) -> nn.Module:
    """
    Train a model on P_perp projected features (OrthoP method).
    """
    input_dim = P.shape[0]
    P_perp = torch.eye(input_dim).to(P.device) - P
    
    # Project features (move to CPU for projection)
    P_perp_cpu = P_perp.cpu()
    train_features_proj = (P_perp_cpu @ train_features.t()).t()
    val_features_proj = (P_perp_cpu @ val_features.t()).t()
    
    # Create dataloaders
    train_dataset = FeatureDataset(train_features_proj, train_labels)
    val_dataset = FeatureDataset(val_features_proj, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create and train model
    model_proj = OneHiddenLayerNet(config.feature_dim, config.hidden_dim, 
                                   config.num_classes, config.regime).to(config.device)
    
    print("\nTraining OrthoP model on P_perp features...")
    history = train_model(model_proj, train_loader, val_loader, config)
    
    return model_proj


def plot_results(history: Dict, config: Config):
    """Plot training history and effective rank evolution"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot effective rank
    axes[2].plot(history['effective_rank'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Effective Rank')
    axes[2].set_title('Evolution of Effective Rank')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(config.output_dir) / f'training_history_{config.regime}.png', dpi=300)
    plt.close()

# ----------------------------------------------------------------------
# Extra: Save effective-rank curve to the current directory
# ----------------------------------------------------------------------
def save_effective_rank_curve(history, filename="effective_rank_plot.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(history["effective_rank"], label="Effective Rank")
    plt.xlabel("Epoch")
    plt.ylabel("Effective Rank")
    plt.title("Effective Rank over Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Saved] {filename}")


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
    
    print("="*80)
    print("Low-Dimensional Simplicity Bias (LD-SB) Experiments")
    print(f"Regime: {config.regime}")
    print(f"Device: {config.device}")
    print("="*80)
    
    # Load and extract features
    print("\n1. Loading Imagenette dataset and extracting ResNet-50 features...")
    train_features, train_labels = load_imagenette_with_features(config, split="train")
    val_features, val_labels = load_imagenette_with_features(config, split="val")
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")
    
    # Create dataloaders
    train_dataset = FeatureDataset(train_features, train_labels)
    val_dataset = FeatureDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Train initial model
    print(f"\n2. Training one-hidden layer network ({config.regime} regime)...")
    model = OneHiddenLayerNet(config.feature_dim, config.hidden_dim, 
                              config.num_classes, config.regime).to(config.device)
    
    history = train_model(model, train_loader, val_loader, config)
    
    # Plot results
    print("\n3. Generating training plots...")
    plot_results(history, config)
    
    # Find projection matrix P
    print("\n4. Finding projection matrix P...")
    if config.regime == "rich":
        # Determine rank based on effective rank
        final_eff_rank = history['effective_rank'][-1]
        rank = max(1, int(np.ceil(final_eff_rank)))
        print(f"Using rank k={rank} (based on effective rank: {final_eff_rank:.2f})")
        P = find_projection_matrix_rich(model, rank)
    else:  # lazy
        rank = max(1, int(config.feature_dim * 0.01))  # Use 1% of dimensions
        print(f"Using rank k={rank}")
        P = find_projection_matrix_lazy(model, train_features, train_labels, rank, config)
    
    # Evaluate LD-SB
    print("\n5. Evaluating LD-SB metrics...")
    ldsb_metrics = evaluate_ld_sb(model, P, val_features, val_labels, config)
    
    print("\nLD-SB Metrics:")
    print(f"  rank(P): {ldsb_metrics['rank_P']}")
    print(f"  P_perp-LC: {ldsb_metrics['P_perp_LC']:.2f}% (should be low)")
    print(f"  P-LC: {ldsb_metrics['P_LC']:.2f}% (should be high)")
    print(f"  P_perp-pC: {ldsb_metrics['P_perp_pC']:.2f}% (should be low)")
    print(f"  P-pC: {ldsb_metrics['P_pC']:.2f}% (should be high)")
    
    # Train OrthoP model
    print("\n6. Training OrthoP model...")
    model_proj = train_orthop_model(P, train_features, train_labels, 
                                    val_features, val_labels, config)
    
    # Evaluate original and OrthoP models
    print("\n7. Final evaluation...")
    model.eval()
    model_proj.eval()
    
    # Original model accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(config.device), labels.to(config.device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc_original = 100. * correct / total
    
    # OrthoP model accuracy (on P_perp features)
    P_perp = torch.eye(config.feature_dim).to(P.device) - P
    P_perp_cpu = P_perp.cpu()
    val_features_proj = (P_perp_cpu @ val_features.t()).t()
    val_loader_proj = DataLoader(FeatureDataset(val_features_proj, val_labels), 
                                 batch_size=config.batch_size, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader_proj:
            features, labels = features.to(config.device), labels.to(config.device)
            outputs = model_proj(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc_orthop = 100. * correct / total
    
    print(f"\nAccuracy(f): {acc_original:.2f}%")
    print(f"Accuracy(f_proj): {acc_orthop:.2f}%")
    print(f"|Acc(f) - Acc(f_proj)|: {abs(acc_original - acc_orthop):.2f}% (epsilon_2 in Definition 1.1)")
    
    # Save results
    results = {
        'regime': config.regime,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'final_effective_rank': history['effective_rank'][-1],
        'ldsb_metrics': ldsb_metrics,
        'acc_original': acc_original,
        'acc_orthop': acc_orthop,
        'acc_diff': abs(acc_original - acc_orthop)
    }
    
    with open(Path(config.output_dir) / f'results_{config.regime}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {config.output_dir}")
    print("\n" + "="*80)
    print("Experiment completed successfully!")
    print("="*80)
    
    save_effective_rank_curve(history)


if __name__ == "__main__":
    main()