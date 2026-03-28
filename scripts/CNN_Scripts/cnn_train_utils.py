from __future__ import annotations

from pathlib import Path
import json
import random
from collections import Counter
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "length": batch["length"].to(device),
        "label": batch["label"].to(device),
        "doc_id": batch["doc_id"],
        "chunk_id": batch["chunk_id"],
        "split": batch["split"],
        "text": batch["text"],
    }

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

def compute_class_weights(train_dataset) -> torch.Tensor:
    labels = [row["label"] for row in train_dataset.rows]
    counts = Counter(labels)

    total = len(labels)
    num_classes = len(counts)

    weights = []
    for cls in range(num_classes):
        cls_count = counts[cls]
        weight = total / (num_classes * cls_count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        y_true.extend(batch["label"].detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / len(loader)

    metrics = {"loss": avg_loss}
    metrics.update(compute_classification_metrics(y_true, y_pred))

    return metrics

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        y_true.extend(batch["label"].detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / len(loader)

    metrics = {"loss": avg_loss}
    metrics.update(compute_classification_metrics(y_true, y_pred))

    return metrics

def save_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)