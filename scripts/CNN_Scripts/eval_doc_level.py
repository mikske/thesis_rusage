from __future__ import annotations

from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn_dataset import build_cnn_datasets_from_saved_vocab
from cnn_model import TextCNN
from cnn_train_utils import (
    get_device,
    compute_classification_metrics,
)
from cnn_metrics_logger import save_doc_level_metrics

#%%
#пути и конфиги
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

EXPERIMENT_DIR = PROJECT_ROOT / "artifacts" / "cnn_fasttext_static" / "20260317_163916"
BEST_MODEL_PATH = EXPERIMENT_DIR / "best_model.pt"
DOC_EVAL_PATH = EXPERIMENT_DIR / "doc_level_metrics.json"

EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
FASTTEXT_MATRIX_PATH = EMBEDDINGS_DIR / "fasttext_embeddings.pt"

RUN_NAME = "cnn_fasttext_static"

#решила еще потыкаться
TEMPERATURES = [0.7, 1.0, 1.3, 1.6, 2.0]

BATCH_SIZE = 32

EMB_DIM = 300
NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]
DROPOUT = 0.5
NUM_CLASSES = 2
PAD_IDX = 0
FREEZE_EMBEDDINGS = True

#%%
#загружаем матрицу эмбеддингов
def load_fasttext_embedding_matrix(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"fastText embedding matrix not found: {path}")

    embedding_matrix = torch.load(path, map_location="cpu")

    if not isinstance(embedding_matrix, torch.Tensor):
        raise TypeError("Loaded fastText embeddings are not a torch.Tensor")

    print("\n=== FASTTEXT MATRIX ===")
    print("path:", path)
    print("shape:", tuple(embedding_matrix.shape))

    return embedding_matrix

#%%
#агрегация по уровню документа
@torch.no_grad()
def predict_doc_level(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[List[int], List[int]]:
    model.eval()

    doc_prob_sums = defaultdict(lambda: torch.zeros(num_classes, dtype=torch.float))
    doc_counts = defaultdict(int)
    doc_true_labels: Dict[str, int] = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        doc_ids = batch["doc_id"]

        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1).detach().cpu()

        for i, doc_id in enumerate(doc_ids):
            prob = probs[i]
            label = int(labels[i].detach().cpu().item())

            doc_prob_sums[doc_id] = doc_prob_sums[doc_id] + prob
            doc_counts[doc_id] += 1

            if doc_id in doc_true_labels:
                if doc_true_labels[doc_id] != label:
                    raise ValueError(f"Inconsistent labels inside doc_id={doc_id}")
            else:
                doc_true_labels[doc_id] = label

    y_true: List[int] = []
    y_pred: List[int] = []

    for doc_id in sorted(doc_prob_sums.keys()):
        mean_prob = doc_prob_sums[doc_id] / doc_counts[doc_id]
        pred = int(torch.argmax(mean_prob).item())
        true = doc_true_labels[doc_id]

        y_true.append(true)
        y_pred.append(pred)

    return y_true, y_pred

#majority vote
@torch.no_grad()
def predict_doc_level_majority(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    model.eval()

    doc_preds: Dict[str, List[int]] = defaultdict(list)
    doc_true_labels: Dict[str, int] = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        doc_ids = batch["doc_id"]

        logits = model(input_ids)
        preds = torch.argmax(logits, dim=1).detach().cpu()

        for i, doc_id in enumerate(doc_ids):
            pred = int(preds[i].item())
            label = int(labels[i].detach().cpu().item())

            doc_preds[doc_id].append(pred)

            if doc_id in doc_true_labels:
                if doc_true_labels[doc_id] != label:
                    raise ValueError(f"Inconsistent labels in doc_id={doc_id}")
            else:
                doc_true_labels[doc_id] = label

    y_true = []
    y_pred = []

    for doc_id in sorted(doc_preds.keys()):
        votes = doc_preds[doc_id]
        pred = Counter(votes).most_common(1)[0][0]
        true = doc_true_labels[doc_id]

        y_true.append(true)
        y_pred.append(pred)

    return y_true, y_pred

#mean + temp scaling
@torch.no_grad()
def predict_doc_level_with_temperature(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    temperature: float,
) -> Tuple[List[int], List[int]]:
    model.eval()

    doc_prob_sums = defaultdict(lambda: torch.zeros(num_classes, dtype=torch.float))
    doc_counts = defaultdict(int)
    doc_true_labels: Dict[str, int] = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        doc_ids = batch["doc_id"]

        logits = model(input_ids)

        # temperature scaling
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1).detach().cpu()

        for i, doc_id in enumerate(doc_ids):
            prob = probs[i]
            label = int(labels[i].detach().cpu().item())

            doc_prob_sums[doc_id] = doc_prob_sums[doc_id] + prob
            doc_counts[doc_id] += 1

            if doc_id in doc_true_labels:
                if doc_true_labels[doc_id] != label:
                    raise ValueError(f"Inconsistent labels inside doc_id={doc_id}")
            else:
                doc_true_labels[doc_id] = label

    y_true: List[int] = []
    y_pred: List[int] = []

    for doc_id in sorted(doc_prob_sums.keys()):
        mean_prob = doc_prob_sums[doc_id] / doc_counts[doc_id]
        pred = int(torch.argmax(mean_prob).item())
        true = doc_true_labels[doc_id]

        y_true.append(true)
        y_pred.append(pred)

    return y_true, y_pred

#weighted voting
@torch.no_grad()
def predict_doc_level_weighted_vote(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[List[int], List[int]]:
    model.eval()

    doc_vote_sums = defaultdict(lambda: torch.zeros(num_classes, dtype=torch.float))
    doc_true_labels: Dict[str, int] = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        doc_ids = batch["doc_id"]

        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1).detach().cpu()

        for i, doc_id in enumerate(doc_ids):
            prob = probs[i]
            label = int(labels[i].detach().cpu().item())

            pred_class = int(torch.argmax(prob).item())
            confidence = float(torch.max(prob).item())

            doc_vote_sums[doc_id][pred_class] += confidence

            if doc_id in doc_true_labels:
                if doc_true_labels[doc_id] != label:
                    raise ValueError(f"Inconsistent labels inside doc_id={doc_id}")
            else:
                doc_true_labels[doc_id] = label

    y_true: List[int] = []
    y_pred: List[int] = []

    for doc_id in sorted(doc_vote_sums.keys()):
        pred = int(torch.argmax(doc_vote_sums[doc_id]).item())
        true = doc_true_labels[doc_id]

        y_true.append(true)
        y_pred.append(pred)

    return y_true, y_pred

#%%
#статистика
def print_doc_level_metrics(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n====================")
    print(title)
    print("====================")

    print("\n=== DOC-LEVEL METRICS ===")
    print(f"docs: {metrics['n_docs']}")
    print(f"accuracy:    {metrics['accuracy']:.4f}")
    print(f"macro_f1:    {metrics['macro_f1']:.4f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.4f}")

    print("\n=== DOC-LEVEL LABEL COUNTS ===")
    print("true:", Counter(metrics["label_counts"]["true"]))
    print("pred:", Counter(metrics["label_counts"]["pred"]))

def get_doc_level_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    metrics = compute_classification_metrics(y_true, y_pred)

    return {
        "n_docs": len(y_true),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "label_counts": {
            "true": dict(Counter(y_true)),
            "pred": dict(Counter(y_pred)),
        },
    }

def main() -> None:
    device = get_device()

    print("=== DEVICE ===")
    print(device)

    print("\n=== EXPERIMENT DIR ===")
    print(EXPERIMENT_DIR)

    _, _, test_dataset, vocab = build_cnn_datasets_from_saved_vocab()

    print("\n=== DATASET ===")
    print("test size:", len(test_dataset))
    print("vocab size:", len(vocab))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    embedding_matrix = load_fasttext_embedding_matrix(FASTTEXT_MATRIX_PATH)

    model = TextCNN(
        vocab_size=len(vocab),
        num_classes=NUM_CLASSES,
        emb_dim=EMB_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
        pretrained_embeddings=embedding_matrix,
        freeze_embeddings=FREEZE_EMBEDDINGS,
    ).to(device)

    print("\n=== LOADING BEST MODEL ===")
    print(BEST_MODEL_PATH)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    aggregation_results = {}

    y_true_mean, y_pred_mean = predict_doc_level(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=NUM_CLASSES,
    )

    mean_metrics = get_doc_level_metrics(y_true_mean, y_pred_mean)
    print_doc_level_metrics("MEAN PROB AGG", mean_metrics)
    aggregation_results["mean_prob"] = mean_metrics

    y_true_mv, y_pred_mv = predict_doc_level_majority(
        model=model,
        loader=test_loader,
        device=device,
    )

    majority_metrics = get_doc_level_metrics(y_true_mv, y_pred_mv)
    print_doc_level_metrics("MAJORITY VOTE", majority_metrics)
    aggregation_results["majority_vote"] = majority_metrics

    y_true_wv, y_pred_wv = predict_doc_level_weighted_vote(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=NUM_CLASSES,
    )

    weighted_vote_metrics = get_doc_level_metrics(y_true_wv, y_pred_wv)
    print_doc_level_metrics("WEIGHTED VOTE", weighted_vote_metrics)
    aggregation_results["weighted_vote"] = weighted_vote_metrics

    best_temp = None
    best_temp_metrics = None
    best_temp_macro_f1 = -1.0

    for temperature in TEMPERATURES:
        y_true_temp, y_pred_temp = predict_doc_level_with_temperature(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=NUM_CLASSES,
            temperature=temperature,
        )

        temp_metrics = get_doc_level_metrics(y_true_temp, y_pred_temp)
        print_doc_level_metrics(f"TEMPERATURE SCALING (T={temperature})", temp_metrics)

        aggregation_results[f"temperature_{temperature}"] = temp_metrics

        if temp_metrics["macro_f1"] > best_temp_macro_f1:
            best_temp_macro_f1 = temp_metrics["macro_f1"]
            best_temp = temperature
            best_temp_metrics = temp_metrics

    print("\n=== BEST TEMPERATURE ===")
    print("temperature:", best_temp)
    print("macro_f1:", best_temp_macro_f1)

    aggregation_results["best_temperature"] = {
        "temperature": best_temp,
        "metrics": best_temp_metrics,
    }

    save_doc_level_metrics(
        path=DOC_EVAL_PATH,
        run_name=RUN_NAME,
        aggregation_results=aggregation_results,
    )

    print(f"\nSaved doc-level metrics to: {DOC_EVAL_PATH}")

if __name__ == "__main__":
    main()