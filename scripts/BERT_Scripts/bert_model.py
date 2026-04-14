from __future__ import annotations

import argparse
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

#%%
#аргументы
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="cointegrated/rubert-tiny2")
    parser.add_argument("--num_labels", type=int, default=2)

    #бейзлайн режимы
    parser.add_argument("--freeze_encoder", type=int, default=0)

    return parser.parse_args()

#%%
#загружаем бейзлайн конфиг
# %%
def build_config(
    model_name: str,
    num_labels: int = 2,
):
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    return config

#%%
#собираем бейзлайн модель: cointegrated/rubert-tiny2, classification head выход по num_labels=2
def build_baseline_model(
    model_name: str,
    num_labels: int = 2,
):
    config = build_config(model_name=model_name, num_labels=num_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    return model

#%%
#для подсчета весов оборачиваем это все красиво
class WeightedBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        freeze_encoder_flag: int = 0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        config = build_config(model_name=model_name, num_labels=num_labels)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )

        # если head-only, замораживаем encoder
        if freeze_encoder_flag == 1:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        # сохраняем веса классов внутри модели
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # стандартный loss HF не используем
            **kwargs,
        )

        logits = outputs.logits
        loss = None

        if labels is not None:
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }


#%%
#здесь у нас заморзка энкодера, если хотим head only режим. будет обучаться только классификатор
def freeze_encoder(model) -> None:
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

#%%
#здесь у нас бейзлайн с обучаемыми параметрами
def unfreeze_all(model) -> None:
    for param in model.parameters():
        param.requires_grad = True

#%%
#считаем сколько всего параметров, сколько обучаемых, сколько замороженых
def count_parameters(model) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }

#%%
#краткая статистика
def print_model_stats(model) -> None:
    stats = count_parameters(model)

    print("\n=== MODEL STATS ===")
    print(f"total params:     {stats['total_params']:,}")
    print(f"trainable params: {stats['trainable_params']:,}")
    print(f"frozen params:    {stats['frozen_params']:,}")

#%%
#основная функция сбора модели
def build_model(
    model_name: str,
    num_labels: int = 2,
    freeze_encoder_flag: int = 0,
    class_weights: torch.Tensor | None = None,
):
    model = build_baseline_model(
        model_name=model_name,
        num_labels=num_labels,
    )

    if freeze_encoder_flag == 1:
        for param in model.base_model.parameters():
            param.requires_grad = False

    return model

#%%
#sanity check как всегда
def main():
    args = parse_args()

    model = build_model(
        model_name=args.model_name,
        num_labels=args.num_labels,
        freeze_encoder_flag=args.freeze_encoder,
    )

    print_model_stats(model)

    print("\n=== MODEL HEAD ===")
    print(model.classifier)

#%%
if __name__ == "__main__":
    main()