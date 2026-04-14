from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
)

from scripts.BERT_Scripts.bert_dataset import build_bert_datasets
from scripts.BERT_Scripts.bert_model import build_model
from scripts.BERT_Scripts.bert_metrics import compute_classification_metrics
from scripts.BERT_Scripts.bert_train_utils import create_experiment_dir, set_seed, build_run_config
from scripts.BERT_Scripts.bert_metrics_logger import save_json, save_chunk_level_metrics

#%%
#обработка аргументов, специально запрашиваются, чтобы более гибко настраивать обучение
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

#%%
#фиксируем сид, формируем папку эксперимента и сохраняем конфиг запуска
def main():
    args = parse_args()

    set_seed(args.seed)

    input_path = Path(args.input_path)
    project_root = Path(args.project_root).resolve()

    run_name = resolve_run_name(args)

    experiment_dir, best_model_dir, metrics_path, config_path = create_experiment_dir(
        project_root=project_root,
        run_name=run_name,
    )

    run_config = build_run_config(
        input_path=str(input_path),
        project_root=str(project_root),
        run_name=run_name,
        model_name=args.model_name,
        max_length=args.max_length,
        num_labels=args.num_labels,
        freeze=args.freeze,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    save_json(config_path, run_config)

    #%%
    #датасеты и модель
    train_dataset, tokenizer = build_bert_datasets(
        json_path=input_path,
        model_name=args.model_name,
        split="train",
        max_length=args.max_length,
    )

    val_dataset, tokenizer = build_bert_datasets(
        json_path=input_path,
        model_name=args.model_name,
        split="valid",
        max_length=args.max_length,
    )

    model = build_model(
        model_name=args.model_name,
        num_labels=args.num_labels,
        freeze_encoder_flag=args.freeze,
    )

    print(f"[INFO] run_name: {run_name}")
    print(f"[INFO] train size: {len(train_dataset)}")
    print(f"[INFO] val size: {len(val_dataset)}")
    print(f"[INFO] freeze: {args.freeze}")
    print(f"[INFO] experiment_dir: {experiment_dir}")

    #%%
    #переходим к аргументам тренирвоки
    training_args = TrainingArguments(
        output_dir=str(experiment_dir / "trainer_output"),
        overwrite_output_dir=True,

        #оцениваем раз в эпоху, сохраняем раз в эпоху
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,

        #оставляем лучшую по f1_macro
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=1,
        report_to="none",
        seed=args.seed,
    )

    #%%
    #собираем тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics_fn(),
    )

    #%%
    #считаем финальные val метрики, сохраняем токенайзер, лучшую модель и метрики
    train_result = trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=val_dataset)

    trainer.save_metrics(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    chunk_metrics = {
        k: float(v) for k, v in eval_metrics.items()
        if isinstance(v, (int, float))
    }

    save_chunk_level_metrics(
        path=metrics_path,
        run_name=run_name,
        split_name="val",
        metrics=chunk_metrics,
    )

    print(f"[INFO] run_name: {run_name}")
    print(f"[INFO] train size: {len(train_dataset)}")
    print(f"[INFO] val size: {len(val_dataset)}")
    print(f"[INFO] freeze: {args.freeze}")
    print(f"[INFO] experiment_dir: {experiment_dir}")

#%%
if __name__ == "__main__":
    main()