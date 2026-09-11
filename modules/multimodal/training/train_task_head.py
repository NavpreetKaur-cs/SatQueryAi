import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from modules.multimodal.task_head import LandCoverHead


def per_class_f1(preds, targets):
    tp = ((preds == 1) & (targets == 1)).sum(dim=0).float()
    fp = ((preds == 1) & (targets == 0)).sum(dim=0).float()
    fn = ((preds == 0) & (targets == 1)).sum(dim=0).float()
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default="models/multimodal/checkpoints/features_cache.npz")
    parser.add_argument("--output", type=str, default="models/multimodal/checkpoints/land_cover_head.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cache = np.load(args.cache, allow_pickle=True)
    features = torch.from_numpy(cache["features"]).float()
    labels = torch.from_numpy(cache["labels"]).float()
    category_order = list(cache["category_order"])

    print(f"Loaded {features.shape[0]} examples, feature_dim={features.shape[1]}, "
          f"categories={category_order}")

    dataset = TensorDataset(features, labels)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Per-class positive weight computed from the TRAIN split only,
    # so validation stays untouched by this. weight = (#neg / #pos) per
    # category -- upweights the loss contribution of rare-positive
    # categories like built_up so the model can't just always say "no".
    train_labels = labels[train_ds.indices]
    pos_counts = train_labels.sum(dim=0).clamp(min=1)
    neg_counts = train_labels.shape[0] - pos_counts
    pos_weight = (neg_counts / pos_counts)
    print(f"Per-class pos_weight (train split): "
          f"{dict(zip(category_order, [round(w, 2) for w in pos_weight.tolist()]))}")

    model = LandCoverHead(input_dim=features.shape[1], categories=category_order)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCELoss(reduction="none")

    def weighted_loss(preds, targets):
        raw = bce(preds, targets)
        weight = 1.0 + (pos_weight - 1.0) * targets  # pos_weight for y=1, 1.0 for y=0
        return (raw * weight).mean()

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = weighted_loss(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                loss = weighted_loss(preds, y)
                val_loss += loss.item() * x.size(0)
                all_preds.append((preds >= 0.5).float())
                all_targets.append(y)
        val_loss /= len(val_ds) if len(val_ds) > 0 else 1

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        precision, recall, f1 = per_class_f1(all_preds, all_targets)
        macro_f1 = f1.mean().item()

        overall_acc = (all_preds == all_targets).float().mean().item()
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={overall_acc:.4f} macro_f1={macro_f1:.4f}")
        if epoch == args.epochs or epoch % 5 == 0:
            for i, cat in enumerate(category_order):
                print(f"    {cat:<12} precision={precision[i]:.3f} recall={recall[i]:.3f} f1={f1[i]:.3f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = model.state_dict()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    print(f"\nBest macro_f1={best_macro_f1:.4f}. Saved best checkpoint to {output_path}")


if __name__ == "__main__":
    main()
