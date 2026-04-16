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


#%%
#собираем предсказания по чанкам
def collect_chunk_predictions(
    model,
    dataset: BERTDataset,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    rows: List[Dict[str, Any]] = []

    sample_offset = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            batch_size_actual = len(labels)

            for i in range(batch_size_actual):
                sample_meta = dataset.samples[sample_offset + i]

                rows.append(
                    {
                        "doc_id": sample_meta["doc_id"],
                        "chunk_id": sample_meta["chunk_id"],
                        "true_label": int(labels[i]),
                        "pred_label": int(preds[i]),
                        "prob_class_0": float(probs[i][0]),
                        "prob_class_1": float(probs[i][1]),
                    }
                )

            sample_offset += batch_size_actual

    return rows


#
#делаем меджорити воут
def aggregate_majority_vote(
    chunk_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)

    for row in chunk_rows:
        grouped[row["doc_id"]].append(row)

    doc_rows: List[Dict[str, Any]] = []

    for doc_id, rows in grouped.items():
        true_labels = {row["true_label"] for row in rows}
        if len(true_labels) != 1:
            raise ValueError(f"Document {doc_id} has inconsistent true labels: {true_labels}")

        true_label = rows[0]["true_label"]
        pred_counts = Counter(row["pred_label"] for row in rows)

        pred_label = pred_counts.most_common(1)[0][0]

        doc_rows.append(
            {
                "doc_id": doc_id,
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "n_chunks": len(rows),
            }
        )

    return doc_rows


#%%
#делаем мин пробабилити
def aggregate_mean_proba(
    chunk_rows: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)

    for row in chunk_rows:
        grouped[row["doc_id"]].append(row)

    doc_rows: List[Dict[str, Any]] = []

    for doc_id, rows in grouped.items():
        true_labels = {row["true_label"] for row in rows}
        if len(true_labels) != 1:
            raise ValueError(f"Document {doc_id} has inconsistent true labels: {true_labels}")

        true_label = rows[0]["true_label"]
        mean_prob_class_1 = float(np.mean([row["prob_class_1"] for row in rows]))
        mean_prob_class_0 = float(np.mean([row["prob_class_0"] for row in rows]))

        pred_label = 1 if mean_prob_class_1 >= threshold else 0

        doc_rows.append(
            {
                "doc_id": doc_id,
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "mean_prob_class_0": mean_prob_class_0,
                "mean_prob_class_1": mean_prob_class_1,
                "n_chunks": len(rows),
            }
        )

    return doc_rows


#%%
#метрики на уровне документов
def evaluate_doc_rows(doc_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    y_true = [row["true_label"] for row in doc_rows]
    y_pred = [row["pred_label"] for row in doc_rows]

    return compute_classification_metrics(y_true=y_true, y_pred=y_pred)


#%%
def main():
    args = parse_args()

    input_path = Path(args.input_path)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    print(f"[INFO] tokenizer vocab size: {len(tokenizer)}")
    print(f"[INFO] model vocab size: {model.get_input_embeddings().num_embeddings}")

    model.to(device)

    dataset = BERTDataset(
        json_path=input_path,
        tokenizer=tokenizer,
        split=args.split,
        max_length=args.max_length,
    )

    print(f"[INFO] split: {args.split}")
    print(f"[INFO] dataset size: {len(dataset)}")
    print(f"[INFO] model_dir: {model_dir}")
    print(f"[INFO] device: {device}")

    chunk_rows = collect_chunk_predictions(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
    )

    majority_doc_rows = aggregate_majority_vote(chunk_rows)
    mean_doc_rows = aggregate_mean_proba(chunk_rows, threshold=0.5)

    majority_metrics = evaluate_doc_rows(majority_doc_rows)
    mean_metrics = evaluate_doc_rows(mean_doc_rows)

    payload = {
        "model_dir": str(model_dir),
        "split": args.split,
        "max_length": args.max_length,
        "n_chunk_rows": len(chunk_rows),
        "n_docs": len(majority_doc_rows),
        "majority_vote": {
            "metrics": majority_metrics,
            "doc_rows": majority_doc_rows,
        },
        "mean_proba": {
            "metrics": mean_metrics,
            "doc_rows": mean_doc_rows,
        },
    }

    save_json(output_path, payload)

    print("\n=== DOC-LEVEL RESULTS ===")
    print("[majority_vote]")
    print(majority_metrics)

    print("\n[mean_proba]")
    print(mean_metrics)

    print(f"\nSaved results to: {output_path}")


#%%
if __name__ == "__main__":
    main()