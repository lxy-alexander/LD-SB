from wilds import get_dataset
import os

# Download only if not already present
root_dir = "./data"
dataset = get_dataset(
    dataset="waterbirds", 
    download=not os.path.exists(os.path.join(root_dir, "waterbirds_v1.0")), 
    root_dir=root_dir
)

# Get splits
train_data = dataset.get_subset("train")
val_data = dataset.get_subset("val")
test_data = dataset.get_subset("test")

print(f"Train size: {len(train_data)}")
print(f"Val size: {len(val_data)}")
print(f"Test size: {len(test_data)}")