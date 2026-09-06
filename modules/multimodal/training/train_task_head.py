"""
modules/multimodal/training/train_task_head.py

Trains LandCoverHead on the feature cache produced by
extract_features_cache.py, with a held-out validation split and
validation-based checkpoint selection (saves the best epoch, not just
the last one).

Run from the repo root:
    python -m modules.multimodal.training.train_task_head
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from modules.multimodal.task_head import LandCoverHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=str, default="models/multimodal/checkpoints/features_cache.npz",
    )
    parser.add_argument(
        "--output", type=str, default="models/multimodal/checkpoints/land_cover_head.pt",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    cache = np.load(args.cache, allow_pickle=True)
    features = torch.from_numpy(cache["features"]).float()
    labels = torch.from_numpy(cache["labels"]).float()
    category_order = list(cache["category_order"])

    print(f"Loaded {features.shape[0]} examples, feature_dim={features.shape[1]}, "
          f"categories={category_order}")

    dataset = TensorDataset(features, labels)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = LandCoverHead(input_dim=features.shape[1], categories=category_order)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)
                predicted = (preds >= 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.numel()
        val_loss /= len(val_ds) if len(val_ds) > 0 else 1
        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    print(f"\nBest val_acc={best_val_acc:.4f}. Saved best checkpoint to {output_path}")


if __name__ == "__main__":
    main()