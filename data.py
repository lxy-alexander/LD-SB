# data.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import os


class FeatureDataset(Dataset):
    """
    A simple dataset wrapper for precomputed ResNet features.

    Args:
        features (Tensor): Pre-extracted feature tensor of shape (N, D).
        labels (Tensor): Corresponding class labels of shape (N,).

    Returns batches of (feature_vector, label).
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_resnet_features(config, split="train"):
    """
    Extract features using a pretrained ResNet-50 (ImageNet-1K weights).

    Steps:
        1. Load the dataset (train/val) from ImageFolder structure.
        2. Apply standard ImageNet preprocessing.
        3. Pass images through ResNet-50 up to the penultimate layer.
        4. Save flattened 2048-dim feature vectors.

    Args:
        config: Experiment configuration object.
        split  : "train" or "val".

    Returns:
        all_features (Tensor): Shape (N, 2048)
        all_labels   (Tensor): Shape (N,)
    """

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset using ImageFolder structure
    dataset = torchvision.datasets.ImageFolder(root=str(
        Path(config.data_root) / split),
                                               transform=transform)

    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=4)

    # Load pretrained ResNet-50 and remove the final FC layer
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.to(config.device).eval()

    all_features, all_labels = [], []

    print(f"Extracting features from {split} ...")

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(config.device)

            # (B, 2048, 1, 1)
            feats = feature_extractor(images)

            # Flatten to (B, 2048)
            feats = feats.view(images.size(0), -1)

            all_features.append(feats.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def load_cached_or_extract(config, split):
    """
    Load cached ResNet features if available.
    Otherwise, compute and save them to disk.

    Args:
        config: Experiment configuration.
        split : "train" or "val".

    Returns:
        features (Tensor): Shape (N, 2048)
        labels   (Tensor): Shape (N,)
    """

    cache_path = Path(config.output_dir) / f"{split}_features.pt"

    if cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        data = torch.load(cache_path)
        return data["features"], data["labels"]

    # Compute features and save them
    features, labels = extract_resnet_features(config, split)

    torch.save({"features": features, "labels": labels}, cache_path)
    print(f"Saved cached features: {cache_path}")

    return features, labels


def create_feature_loaders(config):
    """
    Create dataloaders for the precomputed feature tensors.

    Returns:
        train_loader: DataLoader for training features.
        val_loader  : DataLoader for validation features.
        val_x       : Raw validation feature tensor (for LD-SB evaluation).
        val_y       : Raw validation label tensor.
    """

    # Load or extract feature tensors
    train_x, train_y = load_cached_or_extract(config, "train")
    val_x, val_y = load_cached_or_extract(config, "val")

    # Wrap into PyTorch dataloaders
    train_loader = DataLoader(FeatureDataset(train_x, train_y),
                              batch_size=config.batch_size,
                              shuffle=True)

    val_loader = DataLoader(FeatureDataset(val_x, val_y),
                            batch_size=config.batch_size,
                            shuffle=False)

    return train_loader, val_loader, val_x, val_y
