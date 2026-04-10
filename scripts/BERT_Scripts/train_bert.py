from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
)

from bert_dataset import build_bert_datasets
from bert_model import build_model
from bert_metrics import compute_classification_metrics
from bert_train_utils import create_experiment_dir, set_seed, build_run_config
from bert_metrics_logger import save_json, save_chunk_level_metrics

#%%
#обработка аргументов
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="cointegrated/rubert-tiny2")

    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

#%%
#compute_metrics для тренера
def build_compute_metrics_fn():
    def compute_metrics(eval_pred) -> Dict[str, Any]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        metrics = compute_classification_metrics(
            y_true=labels.tolist(),
            y_pred=preds.tolist()
        )
        return metrics

    return compute_metrics

#%%
#здесь у нас происходит сборка run_name
def resolve_run_name(args) -> str:
    if args.run_name is not None:
        return args.run_name

    freeze_tag = "head_only" if args.freeze == 1 else "full_ft"
    return f"bert_{freeze_tag}_{args.max_length}"
