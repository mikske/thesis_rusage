from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

#%%
#т.к. изначалььно у нас три датасета с разным количеством токенов для экспериментов
#+ эксперименты проводятся в колабе, сначала разбираемся с аргументами командной строки
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="cointegrated/rubert-tiny2")
    parser.add_argument("--max_length", type=int, default=510) #чтоб не падало случайно
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"], default="train")

    return parser.parse_args()

#%%
#читаем jsonl
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in line {line_num}: {e}")

    return rows

class BERTDataset(Dataset):
    def __init__(
            self,
            json_path: Path,
            tokenizer,
            split: str,
            max_length: int = 510,
    ) -> None:
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length

        self.samples: List[Dict[str, Any]] = []
        self._build()

    #%%
    #собираем список всех нужных семплов
    def _build(self) -> None:
        rows = read_jsonl(self.json_path)

        for row in rows:
            if row.get("split") != self.split:
                continue

            text = row.get("text").strip()
            label = row.get("label")
            doc_id = row.get("doc_id")
            chunk_id = row.get("chunk_id")

            if not text:
                continue
            if label is None:
                continue
            if doc_id is None:
                continue
            if chunk_id is None:
                continue

            self.samples.append(
                {
                    "text": text,
                    "label": int(label),
                    "doc_id": str(doc_id), #не идет в модель, но нужно для агрегации документов
                    "chunk_id": int(chunk_id), #не идет в модель, но нужно для агрегации документов
                }
            )

    #%%
    #возвращает количество семплов в дс
    def __len__(self) -> int:
        return len(self.samples)

    #%%
    #берем один чанк и превращаем его в вход для модели
    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }

#%%
#функция сборкт дс: загружает токенизотор, создает дс, возвращает дс и токенизатор
def build_bert_datasets(
    json_path: Path,
    model_name: str,
    split: str,
    max_length: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = BERTDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
    )

    return dataset, tokenizer
#%%
#sanity check
def main():
    args = parse_args()

    json_path = Path(args.input_path)

    dataset, tokenizer = build_bert_datasets(
        json_path=json_path,
        model_name=args.model_name,
        split=args.split,
        max_length=args.max_length,
    )

    print(f"[INFO] split: {args.split}")
    print(f"[INFO] dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("\n=== SAMPLE ===")
        print("input_ids shape:", sample["input_ids"].shape)
        print("attention_mask shape:", sample["attention_mask"].shape)
        print("label:", sample["labels"].item())

        raw_sample = dataset.samples[0]
        print("doc_id:", raw_sample["doc_id"])
        print("chunk_id:", raw_sample["chunk_id"])
        print("text preview:", raw_sample["text"][:300].replace("\n", " "))

#%%
if __name__ == "__main__":
    main()