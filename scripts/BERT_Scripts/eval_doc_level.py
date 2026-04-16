from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.BERT_Scripts.bert_dataset import BERTDataset


#%%
#аргументы
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_path", type=str, required=True)

    return parser.parse_args()


#%%
#обычные метрики классификации
def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    precision_by_class, recall_by_class, f1_by_class, support_by_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    for class_id, (p, r, f1, support) in enumerate(
        zip(precision_by_class, recall_by_class, f1_by_class, support_by_class)
    ):
        metrics[f"precision_class_{class_id}"] = float(p)
        metrics[f"recall_class_{class_id}"] = float(r)
        metrics[f"f1_class_{class_id}"] = float(f1)
        metrics[f"support_class_{class_id}"] = float(support)

    return metrics


#%%
#сохраняем json
def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
