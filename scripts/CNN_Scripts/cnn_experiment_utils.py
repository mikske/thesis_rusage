from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Tuple

#%%
#создание папки эксперимента
def create_experiment_dir(project_root: Path, run_name: str) -> Tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = project_root / "artifacts" / run_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = experiment_dir / "best_model.pt"
    metrics_path = experiment_dir / "metrics.json"

    return experiment_dir, best_model_path, metrics_path