"""
Complete implementation of Table 5: Diversity Metrics
Works with your existing models and features
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


class Config:
    data_root = "./imagenette-160"
    output_dir = "./outputs"
    feature_dim = 2048
    hidden_dim = 512
    num_classes = 10
    batch_size = 64
    regime = "rich"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OneHiddenLayerNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, regime="rich"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class FeatureDataset(torch.utils.data.Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_features_if_needed(config):
    """Load or extract validation features"""
    features_path = Path(config.output_dir) / "val_features.pt"
    labels_path = Path(config.output_dir) / "val_labels.pt"

    if features_path.exists() and labels_path.exists():
        print("Loading cached features...")
        features = torch.load(features_path)
        labels = torch.load(labels_path)
        return features, labels

    print("Extracting features from validation set...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_path = Path(config.data_root) / "val"
    dataset = torchvision.datasets.ImageFolder(root=str(val_path),
                                               transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4)

    # Load ResNet-50
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(config.device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(config.device)
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Remap labels to [0, num_classes-1]
    unique_labels = torch.unique(labels)
    label_mapping = {
        old_label.item(): new_label
        for new_label, old_label in enumerate(unique_labels)
    }
    labels_remapped = torch.zeros_like(labels)
    for old_label, new_label in label_mapping.items():
        labels_remapped[labels == old_label] = new_label

    # Save for future use
    torch.save(features, features_path)
    torch.save(labels_remapped, labels_path)

    return features, labels_remapped


def compute_mistake_diversity(model1, model2, features, labels, config):
    """
    Compute Mistake Diversity (Mist-Div):
    
    Mist-Div(f, f̃) = 1 - |{i: f(xi)≠yi & f̃(xi)≠yi}| / min(|{i: f(xi)≠yi}|, |{i: f̃(xi)≠yi}|)
    
    Higher value = models make different mistakes = more diverse
    
    Paper values (Imagenette):
    - (f, f_ind):  3.87%
    - (f, f_proj): 21.15%  ← Much higher!
    """
    dataset = FeatureDataset(features, labels)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=False)

    model1.eval()
    model2.eval()

    mistakes_1 = []
    mistakes_2 = []
    both_mistakes = []

    idx = 0
    with torch.no_grad():
        for feats, labs in dataloader:
            feats, labs = feats.to(config.device), labs.to(config.device)

            pred1 = model1(feats).argmax(dim=1)
            pred2 = model2(feats).argmax(dim=1)

            for i in range(len(labs)):
                wrong1 = (pred1[i] != labs[i]).item()
                wrong2 = (pred2[i] != labs[i]).item()

                if wrong1:
                    mistakes_1.append(idx)
                if wrong2:
                    mistakes_2.append(idx)
                if wrong1 and wrong2:
                    both_mistakes.append(idx)

                idx += 1

    n_mistakes_1 = len(mistakes_1)
    n_mistakes_2 = len(mistakes_2)
    n_both = len(both_mistakes)

    print(f"    Model 1 mistakes: {n_mistakes_1}")
    print(f"    Model 2 mistakes: {n_mistakes_2}")
    print(f"    Both wrong: {n_both}")

    if min(n_mistakes_1, n_mistakes_2) == 0:
        return 100.0  # Perfect accuracy on one model

    mist_div = (1.0 - n_both / min(n_mistakes_1, n_mistakes_2)) * 100

    return mist_div


def compute_cc_logit_correlation(model1, model2, features, labels, config):
    """
    Compute Class-Conditioned Logit Correlation (CC-LogitCorr):
    
    CC-LogitCorr(f, f̃) = (1/|Y|) Σ_y Corr([f(xi)], [f̃(xi)] : yi=y)
    
    Lower value = outputs less correlated = more diverse
    
    Paper values (Imagenette):
    - (f, f_ind):  99.88%
    - (f, f_proj): 90.86%  ← Much lower!
    """
    dataset = FeatureDataset(features, labels)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=False)

    model1.eval()
    model2.eval()

    # Collect logits per class
    logits_per_class_1 = {i: [] for i in range(config.num_classes)}
    logits_per_class_2 = {i: [] for i in range(config.num_classes)}

    with torch.no_grad():
        for feats, labs in dataloader:
            feats, labs = feats.to(config.device), labs.to(config.device)

            logits1 = model1(feats)
            logits2 = model2(feats)

            for i in range(len(labs)):
                class_id = labs[i].item()
                logits_per_class_1[class_id].append(logits1[i].cpu().numpy())
                logits_per_class_2[class_id].append(logits2[i].cpu().numpy())

    # Compute correlation for each class
    correlations = []

    for class_id in range(config.num_classes):
        if len(logits_per_class_1[class_id]) < 2:
            continue

        logits1 = np.array(
            logits_per_class_1[class_id])  # (n_samples, num_classes)
        logits2 = np.array(logits_per_class_2[class_id])

        # Flatten to compute correlation
        logits1_flat = logits1.flatten()
        logits2_flat = logits2.flatten()

        # Compute Pearson correlation
        if len(logits1_flat) > 1:
            corr = np.corrcoef(logits1_flat, logits2_flat)[0, 1]
            correlations.append(corr)
            print(f"    Class {class_id}: correlation = {corr:.4f}")

    # Average across classes
    cc_logit_corr = np.mean(correlations) * 100

    return cc_logit_corr


def main():
    config = Config()

    print("=" * 80)
    print("TABLE 5: DIVERSITY METRICS")
    print("=" * 80)

    # Load features
    print("\n1. Loading validation features...")
    val_features, val_labels = load_features_if_needed(config)
    print(f"   Loaded {len(val_features)} validation samples")

    # Load models
    print("\n2. Loading models...")

    # Model f (original)
    model_f = OneHiddenLayerNet(config.feature_dim, config.hidden_dim,
                                config.num_classes,
                                config.regime).to(config.device)
    f_path = f"{config.output_dir}/best_model_{config.regime}.pth"
    if not Path(f_path).exists():
        print(f"ERROR: Model f not found at {f_path}")
        return
    model_f.load_state_dict(torch.load(f_path))
    model_f.eval()
    print("   ✓ Loaded model f")

    # Model f_ind (independent)
    model_ind = None
    ind_path = f"{config.output_dir}/best_model_{config.regime}_ind.pth"
    if Path(ind_path).exists():
        model_ind = OneHiddenLayerNet(config.feature_dim, config.hidden_dim,
                                      config.num_classes,
                                      config.regime).to(config.device)
        model_ind.load_state_dict(torch.load(ind_path))
        model_ind.eval()
        print("   ✓ Loaded model f_ind")
    else:
        print("   ✗ Model f_ind not found")

    # Model f_proj (OrthoP)
    model_proj = None
    proj_path = f"{config.output_dir}/best_model_{config.regime}_proj.pth"
    if Path(proj_path).exists():
        model_proj = OneHiddenLayerNet(config.feature_dim, config.hidden_dim,
                                       config.num_classes,
                                       config.regime).to(config.device)
        model_proj.load_state_dict(torch.load(proj_path))
        model_proj.eval()
        print("   ✓ Loaded model f_proj")
    else:
        print("   ✗ Model f_proj not found (run main experiment first)")

    # Compute metrics
    results = {}

    print("\n" + "=" * 80)
    print("COMPUTING DIVERSITY METRICS")
    print("=" * 80)

    if model_ind is not None:
        print("\n3a. Metrics for (f, f_ind)...")
        print("-" * 80)

        print("  Computing Mistake Diversity...")
        mist_div_ind = compute_mistake_diversity(model_f, model_ind,
                                                 val_features, val_labels,
                                                 config)

        print("  Computing CC-Logit Correlation...")
        cc_corr_ind = compute_cc_logit_correlation(model_f, model_ind,
                                                   val_features, val_labels,
                                                   config)

        results['f_vs_find'] = {
            'Mist-Div': mist_div_ind,
            'CC-LogitCorr': cc_corr_ind
        }

        print(f"\n  Results:")
        print(f"    Mistake Diversity:      {mist_div_ind:.2f}%")
        print(f"    CC-Logit Correlation:   {cc_corr_ind:.2f}%")

    if model_proj is not None:
        print("\n3b. Metrics for (f, f_proj)...")
        print("-" * 80)

        print("  Computing Mistake Diversity...")
        mist_div_proj = compute_mistake_diversity(model_f, model_proj,
                                                  val_features, val_labels,
                                                  config)

        print("  Computing CC-Logit Correlation...")
        cc_corr_proj = compute_cc_logit_correlation(model_f, model_proj,
                                                    val_features, val_labels,
                                                    config)

        results['f_vs_fproj'] = {
            'Mist-Div': mist_div_proj,
            'CC-LogitCorr': cc_corr_proj
        }

        print(f"\n  Results:")
        print(f"    Mistake Diversity:      {mist_div_proj:.2f}%")
        print(f"    CC-Logit Correlation:   {cc_corr_proj:.2f}%")

    # Comparison table
    if model_ind is not None and model_proj is not None:
        print("\n" + "=" * 80)
        print("TABLE 5: COMPARISON")
        print("=" * 80)

        print(
            f"\n{'Metric':<30} {'(f, f_ind)':<18} {'(f, f_proj)':<18} {'Expected':<25}"
        )
        print("-" * 95)
        print(
            f"{'Mist-Div (%)':<30} {mist_div_ind:>17.2f} {mist_div_proj:>17.2f} {'f_proj > f_ind':<25}"
        )
        print(
            f"{'CC-LogitCorr (%)':<30} {cc_corr_ind:>17.2f} {cc_corr_proj:>17.2f} {'f_proj < f_ind':<25}"
        )

        print("\n" + "=" * 80)
        print("PAPER VALUES (Table 5, B-Imagenette)")
        print("=" * 80)
        print("Mist-Div:")
        print("  (f, f_ind):  3.87 ± 1.54%")
        print("  (f, f_proj): 21.15 ± 1.57%  ← 5.5x higher!")
        print("\nCC-LogitCorr:")
        print("  (f, f_ind):  99.88 ± 0.01%")
        print("  (f, f_proj): 90.86 ± 1.08%  ← 9% lower!")

        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)

        mist_div_good = mist_div_proj > mist_div_ind
        cc_corr_good = cc_corr_proj < cc_corr_ind

        if mist_div_good:
            print("✓ Mistake Diversity: f_proj makes MORE DIFFERENT mistakes")
            print(f"   {mist_div_proj:.1f}% vs {mist_div_ind:.1f}% "
                  f"({(mist_div_proj/mist_div_ind - 1)*100:.0f}% improvement)")
        else:
            print("✗ Mistake Diversity: f_proj not more diverse")

        if cc_corr_good:
            print("✓ CC-Logit Correlation: f_proj outputs LESS CORRELATED")
            print(f"   {cc_corr_proj:.1f}% vs {cc_corr_ind:.1f}% "
                  f"({cc_corr_ind - cc_corr_proj:.1f}% reduction)")
        else:
            print("✗ CC-Logit Correlation: f_proj not less correlated")

        if mist_div_good and cc_corr_good:
            print("\n" + " " * 20)
            print("SUCCESS: OrthoP creates TRULY DIVERSE models!")
            print("         This validates the LD-SB hypothesis.")
            print(" " * 20)
        else:
            print("\n  Results don't fully match expectations")
            print("   Possible reasons:")
            print("   - OrthoP training may not have converged")
            print("   - rank(P) choice may not be optimal")
            print("   - Dataset differences")

    elif model_ind is not None:
        print(
            "\n  Only f_ind available. Complete main experiment to get f_proj."
        )
    elif model_proj is not None:
        print(
            "\n  Only f_proj available. Train f_ind to get complete comparison."
        )

    # Save results
    output_file = Path(config.output_dir) / "diversity_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
