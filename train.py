# train.py
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from utils import compute_effective_rank

def train_model(model, train_loader, val_loader, config):

    criterion = torch.nn.CrossEntropyLoss()
    lr = config.learning_rate_rich if config.regime == "rich" else config.learning_rate_lazy
    
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=config.momentum,
                          weight_decay=config.weight_decay)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.num_steps - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "effective_rank"]}

    # Compute initial rank
    eff_rank = compute_effective_rank(model.get_first_layer_weights())

    step = 0
    train_iter = iter(train_loader)

    while step < config.num_steps:
        model.train()

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(config.device), y.to(config.device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        # Evaluate once per epoch
        if step % len(train_loader) == 0:
            val_loss, val_acc = evaluate(model, val_loader, config)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            eff_rank = compute_effective_rank(model.get_first_layer_weights())
            history["effective_rank"].append(eff_rank)

            print(f"Step {step}/{config.num_steps} | Val Acc {val_acc:.2f}% | Rank {eff_rank:.2f}")

    return history


def evaluate(model, loader, config):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total, correct, total_loss = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), 100 * correct / total
