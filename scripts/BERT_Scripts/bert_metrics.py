# %%
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight


#%%
#считаем базовые метрики
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

    return metrics


#%%
#считаем веса классов по лейблам
def compute_class_weights(
    y_train: List[int],
) -> Dict[int, float]:
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(y_train),
    )

    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


#%%
def main():
    y_true = [0, 1, 0, 1, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 1, 0, 1]

    metrics = compute_classification_metrics(y_true, y_pred)
    print("=== BASIC METRICS ===")
    print(metrics)

    class_weights = compute_class_weights(y_true)
    print("\n=== CLASS WEIGHTS ===")
    print(class_weights)


#%%
if __name__ == "__main__":
    main()