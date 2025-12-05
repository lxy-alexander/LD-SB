import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import os

import pandas as pd
from PIL import Image


class WaterbirdsDataset(Dataset):
    # Dataset loader for Waterbirds using metadata.csv.
    # Returns (image_tensor, label).

    def __init__(self, root, split, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        metadata_path = self.root / "metadata.csv"
        df = pd.read_csv(metadata_path)

        # Mapping: 0=train, 1=val, 2=test
        split_map = {"train": 0, "val": 1, "test": 2}
        df = df[df["split"] == split_map[split]]

        # Store paths and labels
        self.img_paths = df["img_filename"].tolist()
        self.labels = df["y"].tolist()  # 0=landbird, 1=waterbird

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.root / self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


class FeatureDataset(Dataset):
    # Dataset wrapper for precomputed ResNet feature tensors.
    # Returns (feature_vector, label).

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_resnet_features(config, split="train"):
    # Extract pretrained ResNet-50 features for the given split.

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load Waterbirds dataset (not ImageFolder)
    dataset = WaterbirdsDataset(config.data_root, split, transform)

    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=4)

    # Load pretrained ResNet-50 and remove the classifier
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.to(config.device).eval()

    all_features, all_labels = [], []

    print(f"Extracting features from Waterbirds split={split} ...")

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(config.device)

            feats = feature_extractor(images)  # shape: (N, 2048, 1, 1)
            feats = feats.view(images.size(0), -1)  # flatten to (N, 2048)

            all_features.append(feats.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def load_cached_or_extract(config, split):
    # Load saved features if available; otherwise extract and cache.

    cache_path = Path(config.output_dir) / f"{split}_features.pt"

    if cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        data = torch.load(cache_path)
        return data["features"], data["labels"]

    # Extract features and save to disk
    features, labels = extract_resnet_features(config, split)
    torch.save({"features": features, "labels": labels}, cache_path)

    print(f"Saved cached features: {cache_path}")
    return features, labels


def create_feature_loaders(config):
    # Create dataloaders for precomputed feature tensors.
    # Returns: train_loader, val_loader, raw_val_features, raw_val_labels

    train_x, train_y = load_cached_or_extract(config, "train")
    val_x, val_y = load_cached_or_extract(config, "val")

    train_loader = DataLoader(FeatureDataset(train_x, train_y),
                              batch_size=config.batch_size,
                              shuffle=True)

    val_loader = DataLoader(FeatureDataset(val_x, val_y),
                            batch_size=config.batch_size,
                            shuffle=False)

    return train_loader, val_loader, val_x, val_y
