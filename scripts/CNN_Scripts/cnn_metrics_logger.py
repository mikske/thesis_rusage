from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict

#%%
#базовое сохранение json
def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

#%%
#универсальный логгер для док-евал
def save_doc_level_metrics(
    path: Path,
    run_name: str,
    aggregation_results: Dict[str, Dict[str, Any]],
) -> None:
    payload = {
        "run_name": run_name,
        "doc_level_results": aggregation_results,
    }

    save_json(path, payload)