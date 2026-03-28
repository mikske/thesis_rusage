#%%
from __future__ import annotations

from pathlib import Path
import json
import re
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset

#%%
#готовый датасет после спелита
DATASET_PATH = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/cnn_chunks_with_split.jsonl")

#сюда сохраняем словарь
VOCAB_PATH = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/cnn_vocab.json")

#спец токены
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

#параметры словаря
MIN_FREQ = 2
MAX_VOCAB_SIZE = 30000

#бейзлайн макс последовательность
MAX_SEQ_LEN = 700

#%%
#читаем jsonl
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows

#%%
#для бейзлайна испольую простой токенайзер; слова, числа, пунктцация отдельными токенами
_WORD_RE = re.compile(r"[а-яёa-z0-9]+|[.,!?;:()\"-]", flags=re.IGNORECASE)

def normalize_text_for_cnn(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"<[^>]+>", " ", text)   # убираем <CHAPTER> и подобные теги
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    text = normalize_text_for_cnn(text)
    return _WORD_RE.findall(text)

def count_cnn_tokens(text: str) -> int:
    return len(tokenize_text(text))
#%%
#фильтрация по сплиту
def filter_rows_by_split(
        rows: List[Dict[str, Any]],
        split: str,
) -> List[Dict[str, Any]]:
    return [row for row in rows if row["split"] == split]

#%%
#пострение словаря по трейну
def build_vocab(
    rows: List[Dict[str, Any]],
    min_freq: int = MIN_FREQ,
    max_vocab_size: int = MAX_VOCAB_SIZE,
) -> Dict[str, int]:
    counter: Counter = Counter()

    for i, row in enumerate(rows):
        text = row["text"]
        tokens = tokenize_text(text)

        if i == 0:
            print("\n=== BUILD_VOCAB DEBUG ===")
            print("RAW TEXT:", text[:300])
            print("TOKENS:", tokens[:50])
            print("TOKENS TYPE:", type(tokens))
            if len(tokens) > 0:
                print("FIRST TOKEN TYPE:", type(tokens[0]))

        counter.update(tokens)

    items = [(token, freq) for token, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[: max_vocab_size - 2]

    print("\n=== TOP COUNTER TOKENS ===")
    print(items[:30])

    vocab: Dict[str, int] = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
    }

    for idx, (token, _) in enumerate(items, start=2):
        vocab[token] = idx

    return vocab

#%%
#сохранение и загузка словаря
def save_vocab(path: Path, vocab: Dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        vocab: Dict[str, int] = json.load(f)

    return vocab

#%%
#преобразование текста в ids
def numericalize_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(token, unk_id) for token in tokens]

#%%
#паддинг и транкейшн
#возвращает input_ids фикс длины, real_length до паддинга
def pad_or_truncate(
        token_ids: List[int],
        max_len: int = MAX_SEQ_LEN,
        pad_id: int = PAD_ID,
) -> Tuple[List[int], int]:

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]

    real_length = len(token_ids)

    if len(token_ids) < max_len:
        token_ids = token_ids + [pad_id] * (max_len - len(token_ids))

    return token_ids, real_length

#%%
#датасет для CNN
class CNNDataset(Dataset):
    def __init__(
            self,
            rows: List[Dict[str, Any]],
            vocab: Dict[str, int],
            max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        self.rows = rows
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        text = row["text"]
        label = row["label"]
        doc_id = row["doc_id"]
        chunk_id = row["chunk_id"]
        split = row["split"]

        tokens = tokenize_text(text)
        token_ids = numericalize_tokens(tokens, self.vocab)
        input_ids, real_length = pad_or_truncate(
            token_ids,
            max_len=self.max_seq_len,
            pad_id=self.vocab[PAD_TOKEN],
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(real_length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "split": split,
            "text": text,
        }

#%%
#сборк трайн вал и тест датасетов
#читаем rows, делим split, строим vocab по трейну, создаем датасеты
def build_cnn_datasets(
    dataset_path: Path = DATASET_PATH,
    vocab_path: Path = VOCAB_PATH,
    min_freq: int = MIN_FREQ,
    max_vocab_size: int = MAX_VOCAB_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[CNNDataset, CNNDataset, CNNDataset, Dict[str, int]]:
    rows = read_jsonl(dataset_path)

    train_rows = filter_rows_by_split(rows, "train")
    val_rows = filter_rows_by_split(rows, "val")
    test_rows = filter_rows_by_split(rows, "test")

    vocab = build_vocab(
        rows=train_rows,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )

    save_vocab(vocab_path, vocab)

    train_dataset = CNNDataset(train_rows, vocab=vocab, max_seq_len=max_seq_len)
    val_dataset = CNNDataset(val_rows, vocab=vocab, max_seq_len=max_seq_len)
    test_dataset = CNNDataset(test_rows, vocab=vocab, max_seq_len=max_seq_len)

    return train_dataset, val_dataset, test_dataset, vocab

#%%
#используем уже сохраненный словарь и не пересобираем, нужно для pretrained emb
def build_cnn_datasets_from_saved_vocab(
    dataset_path: Path = DATASET_PATH,
    vocab_path: Path = VOCAB_PATH,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[CNNDataset, CNNDataset, CNNDataset, Dict[str, int]]:

    rows = read_jsonl(dataset_path)

    train_rows = filter_rows_by_split(rows, "train")
    val_rows = filter_rows_by_split(rows, "val")
    test_rows = filter_rows_by_split(rows, "test")

    vocab = load_vocab(vocab_path)

    train_dataset = CNNDataset(train_rows, vocab=vocab, max_seq_len=max_seq_len)
    val_dataset = CNNDataset(val_rows, vocab=vocab, max_seq_len=max_seq_len)
    test_dataset = CNNDataset(test_rows, vocab=vocab, max_seq_len=max_seq_len)

    return train_dataset, val_dataset, test_dataset, vocab

#%%
#статистика словаря и датасетов
def print_dataset_stats(
    train_dataset: CNNDataset,
    val_dataset: CNNDataset,
    test_dataset: CNNDataset,
    vocab: Dict[str, int],
) -> None:
    print("\n=== CNN DATASET STATS ===")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples:   {len(val_dataset)}")
    print(f"test samples:  {len(test_dataset)}")
    print(f"vocab size:    {len(vocab)}")
    print(f"pad id:        {vocab[PAD_TOKEN]}")
    print(f"unk id:        {vocab[UNK_TOKEN]}")

def print_length_stats(rows: List[Dict[str, Any]]) -> None:
    lengths = []

    for row in rows:
        tokens = tokenize_text(row["text"])
        lengths.append(len(tokens))

    lengths.sort()
    n = len(lengths)

    def q(p: float) -> int:
        idx = min(int(p * n), n - 1)
        return lengths[idx]

    print("\n=== TOKEN LENGTH STATS ===")
    print(f"n:    {n}")
    print(f"min:  {lengths[0]}")
    print(f"p50:  {q(0.50)}")
    print(f"p75:  {q(0.75)}")
    print(f"p90:  {q(0.90)}")
    print(f"p95:  {q(0.95)}")
    print(f"max:  {lengths[-1]}")

#%%
#самопроверочка
def preview_sample(dataset: CNNDataset, index: int = 0) -> None:
    item = dataset[index]

    print("\n=== SAMPLE PREVIEW ===")
    print("doc_id:", item["doc_id"])
    print("chunk_id:", item["chunk_id"])
    print("split:", item["split"])
    print("label:", item["label"].item())
    print("length:", item["length"].item())
    print("input_ids shape:", item["input_ids"].shape)
    print("text preview:", item["text"][:300].replace("\n", " "))
    print("first 40 ids:", item["input_ids"][:40].tolist())

def print_truncation_stats(rows: List[Dict[str, Any]], max_seq_len: int) -> None:
    lengths = [len(tokenize_text(row["text"])) for row in rows]
    truncated = sum(1 for x in lengths if x > max_seq_len)

    print("\n=== TRUNCATION STATS ===")
    print(f"max_seq_len: {max_seq_len}")
    print(f"total rows:  {len(lengths)}")
    print(f"truncated:   {truncated}")
    print(f"share:       {truncated / len(lengths):.4f}")

#%%
def main() -> None:
    rows = read_jsonl(DATASET_PATH)

    train_rows = filter_rows_by_split(rows, "train")
    val_rows = filter_rows_by_split(rows, "val")
    test_rows = filter_rows_by_split(rows, "test")

    vocab = build_vocab(rows=train_rows)
    save_vocab(VOCAB_PATH, vocab)

    train_dataset = CNNDataset(train_rows, vocab=vocab, max_seq_len=MAX_SEQ_LEN)
    val_dataset = CNNDataset(val_rows, vocab=vocab, max_seq_len=MAX_SEQ_LEN)
    test_dataset = CNNDataset(test_rows, vocab=vocab, max_seq_len=MAX_SEQ_LEN)

    print_dataset_stats(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        vocab=vocab,
    )

    print_length_stats(train_rows)
    print_truncation_stats(train_rows, MAX_SEQ_LEN)

    print("\n=== VOCAB CHECK ===")
    for token in ["в", "и", "на", "когда", "было", "гномы", ",", "."]:
        print(token, "->", vocab.get(token, UNK_ID))

    preview_sample(train_dataset, index=0)

    print(f"\nVocab saved to: {VOCAB_PATH}")
#%%
if __name__ == "__main__":
    main()