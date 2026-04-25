#!/usr/bin/env python3
"""
Fine-tune ResNet18 (ImageNet) for 4-way garment classification from ImageFolder export.

Expects:
  data_dir/train/{class}/*.png
  data_dir/val/{class}/*.png
  data_dir/label_map.json  (optional; if missing, uses torchvision sorted class names)

Requires: torch torchvision

Example:
  python scripts/garment_classifier/train_garment_classifier.py \\
    --data_dir ./garment_classifier_data --output_dir ./garment_classifier_runs/run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_dataloaders(
    data_dir: Path, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader, list[str]]:
    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            imagenet_norm,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet_norm,
        ]
    )
    train_ds = datasets.ImageFolder(str(data_dir / "train"), train_tf)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), val_tf)
    # ImageFolder.classes is sorted alphabetically — must match label_map.json if provided.
    class_names = train_ds.classes
    if val_ds.classes != class_names:
        raise ValueError("Train/val class folder names differ.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_names


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


def confusion_matrix(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> torch.Tensor:
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            for t, p in zip(targets.view(-1), pred.view(-1)):
                cm[t.long(), p.long()] += 1
    return cm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--head_epochs", type=int, default=5, help="Train classifier head only.")
    p.add_argument("--lr_head", type=float, default=1e-2)
    p.add_argument("--lr_full", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    num_classes = len(class_names)

    # Persist canonical mapping used by ImageFolder (class index -> name is implicit order)
    label_by_index = {i: class_names[i] for i in range(num_classes)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "label_map.json").write_text(
        json.dumps({label_by_index[i]: i for i in range(num_classes)}, indent=2) + "\n"
    )
    (args.output_dir / "class_index_to_name.json").write_text(
        json.dumps(label_by_index, indent=2) + "\n"
    )

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_path = args.output_dir / "classifier_best.pt"

    for epoch in range(args.epochs):
        if epoch < args.head_epochs:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
        else:
            if epoch == args.head_epochs:
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = optim.AdamW(
                    model.parameters(), lr=args.lr_full, weight_decay=args.weight_decay
                )

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy_from_logits(logits, targets)
            n_batches += 1

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        vb = 0
        with torch.inference_mode():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                val_loss += criterion(logits, targets).item()
                val_acc += accuracy_from_logits(logits, targets)
                vb += 1
        val_loss /= max(vb, 1)
        val_acc /= max(vb, 1)

        cm = confusion_matrix(model, val_loader, device, num_classes)
        print(
            f"epoch {epoch+1}/{args.epochs} "
            f"train_loss={running_loss/max(n_batches,1):.4f} "
            f"train_acc={running_acc/max(n_batches,1):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        print("val confusion (rows=true, cols=pred):")
        print(cm.numpy())

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "label_map": {label_by_index[i]: i for i in range(num_classes)},
                    "weights_enum": "IMAGENET1K_V1",
                },
                best_path,
            )
            print(f"  saved best to {best_path} (val_acc={best_val:.4f})")

    print(f"Done. Best val_acc={best_val:.4f} checkpoint={best_path}")


if __name__ == "__main__":
    main()
