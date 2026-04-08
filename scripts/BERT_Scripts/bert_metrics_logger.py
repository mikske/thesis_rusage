from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

#%%
#сохраняем json
def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

#%%
#сохранение метрик на уровне чанков
def save_chunk_level_metrics(
        path: Path,
        run_name: str,
        split_name: str,
        metrics: Dict[str, Any]
) -> None:
    payload = {
        "run_name": run_name,
        "level": "chunk",
        split_name: split_name,
        "metrics": metrics
    }

    save_json(path, payload)

#%%
#сохранение метрик на уровне документов
def save_doc_level_metrics(
        path: Path,
        run_name: str,
        split_name: str,
        aggreation_results: Dict[str, Dict[str, Any]],
) -> None:
    payload = {
        "run_name": run_name,
        "level": "doc",
        split_name: str,
        "doc_level_results": aggreation_results,
    }

    save_json(path, payload)

#%%
#метрики указаны случайные, чтоы проверить работает ли логгер и все ли правильно пишется
def main():
    out_dir = Path("./artifacts/logger_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_metrics_path = out_dir / "chunk_metrics.json"
    doc_metrics_path = out_dir / "doc_metrics.json"

    save_chunk_level_metrics(
        path=chunk_metrics_path,
        run_name="bert_baseline_510",
        split_name="val",
        metrics={
            "accuracy": 0.78,
            "macro_f1": 0.76,
            "weighted_f1": 0.78,
        },
    )

    save_doc_level_metrics(
        path=doc_metrics_path,
        run_name="bert_baseline_510",
        aggregation_results={
            "majority_vote": {
                "accuracy": 0.81,
                "macro_f1": 0.79,
            },
            "mean_proba": {
                "accuracy": 0.82,
                "macro_f1": 0.80,
            },
        },
    )

    print("Saved chunk metrics to:", chunk_metrics_path)
    print("Saved doc metrics to:", doc_metrics_path)

#%%
if __name__ == "__main__":
    main()