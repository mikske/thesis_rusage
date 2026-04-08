from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

#%%
#создаем папку эксперимента конкретного запуска
def create_experiment_dir(
    project_root: Path,
    run_name: str,
) -> Tuple[Path, Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    experiment_dir = project_root / "artifacts" / run_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    best_model_dir = experiment_dir / "best_model"
    metrics_path = experiment_dir / "metrics.json"
    config_path = experiment_dir / "run_config.json"

    return experiment_dir, best_model_dir, metrics_path, config_path

#%%
#фиксируем seed
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#%%
#сборка конфига запуска
# %%
def build_run_config(**kwargs) -> Dict[str, Any]:

    return dict(kwargs)

#%%
#sanity check
def main():
    project_root = Path(".").resolve()
    max_length = 510 #просто дефолт на случай если не передали параметр, это для логирования

    experiment_dir, best_model_dir, metrics_path, config_path = create_experiment_dir(
        project_root=project_root,
        run_name=f"bert_baseline_{max_length}",
    )

    print("experiment_dir:", experiment_dir)
    print("best_model_dir:", best_model_dir)
    print("metrics_path:", metrics_path)
    print("config_path:", config_path)

    config = build_run_config(
        model_name="cointegrated/rubert-tiny2",
        max_length=max_length,
        batch_size=8,
        learning_rate=2e-5,
        epochs=3,
        seed=42,
    )

    print("\nrun_config:")
    print(config)

    set_seed(42)
    print("\nSeed fixed.")

#%%
if __name__ == "__main__":
    main()