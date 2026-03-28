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

from cnn_dataset import build_cnn_datasets
from cnn_model import TextCNN, count_trainable_parameters
from cnn_train_utils import (
    set_seed,
    get_device,
    compute_class_weights,
    train_one_epoch,
    evaluate,
    save_metrics,
)
from cnn_experiment_utils import create_experiment_dir

#%%
#пути и конфигурация
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RUN_NAME = "cnn_baseline"

EXPERIMENT_DIR, BEST_MODEL_PATH, METRICS_PATH = create_experiment_dir(
    project_root=PROJECT_ROOT,
    run_name=RUN_NAME,
)

#%%
#гиперпарамтеры обучения
SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3 #пойдет для Adam
WEIGHT_DECAY = 1e-4 #легкая регуляризация
PATIENCE = 4 #для early stopping

#%%
#гиперпараметры модели
EMB_DIM = 300
NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]
DROPOUT= 0.5
NUM_CLASSES = 2

#%%
#главная функция обучения
def main() -> None:
    set_seed(SEED)
    device = get_device()

    print("=== DEVICE ===")
    print(device)

    print("\n=== EXPERIMENT DIR ===")
    print(EXPERIMENT_DIR)

    train_dataset, val_dataset, test_dataset, vocab = build_cnn_datasets()

    print("\n=== DATASETS ===")
    print("train size:", len(train_dataset))
    print("val size:", len(val_dataset))
    print("test size:", len(test_dataset))
    print("vocab size:", len(vocab))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TextCNN(
        vocab_size=len(vocab),
        num_classes=NUM_CLASSES,
        emb_dim=EMB_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT,
        pad_idx=0,
    ).to(device)

    print("\n=== MODEL ===")
    print(model)
    print("trainable params:", count_trainable_parameters(model))

    class_weights = compute_class_weights(train_dataset).to(device)

    print("\n=== CLASS WEIGHTS ===")
    print(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history: Dict[str, List[Dict[str, float]]] = {
        "train": [],
        "val": [],
    }

    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    #%%
    #цикл по эпохам
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'=' * 20}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'=' * 20}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=NUM_CLASSES,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=NUM_CLASSES,
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print("\n[TRAIN]")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")

        print("\n[VAL]")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")

        #%%
        #early stopping и сохранение лучшей модели
        current_val_macro_f1 = val_metrics["macro_f1"]

        if current_val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = current_val_macro_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"\nSaved new best model to: {BEST_MODEL_PATH}")

        else:
            patience_counter += 1
            print(f"\nNo improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n=== LOADING BEST MODEL ===")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=NUM_CLASSES,
    )

    print("\n[TEST]")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    #%%
    #сохранение результатов
    results = {
        "config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "emb_dim": EMB_DIM,
            "num_filters": NUM_FILTERS,
            "kernel_sizes": KERNEL_SIZES,
            "dropout": DROPOUT,
            "num_classes": NUM_CLASSES,
            "run_name": RUN_NAME,
            "experiment_dir": str(EXPERIMENT_DIR),
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
        "test_metrics": test_metrics,
    }

    save_metrics(METRICS_PATH, results)

    print("\n=== SAVED ===")
    print("best model:", BEST_MODEL_PATH)
    print("metrics:", METRICS_PATH)

if __name__ == "__main__":
    main()