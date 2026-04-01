#%%
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List, Any, Tuple
from collections import Counter

from sklearn.model_selection import train_test_split

#%%
#путь к датасету с чанками
DATASET_PATH = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/cnn_chunks_dataset.jsonl")

#сюда сохраняем резы -- 2 файла
OUTPUT_DIR = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json")
DOC_SPLITS_PATH = OUTPUT_DIR / "cnn_doc_splits.json"
CHUNK_SPLITS_PATH = OUTPUT_DIR / "cnn_chunks_with_split.jsonl"

#пропорции сплита и сид
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

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
#пишем функцию сохранения: jsonl для датасета, json для манифеста
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path: Path, obj: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

#%%
#строим таблицу документов
#из чанков собираем таблицу документов, т.е. doc_id -> запись

def build_doc_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = {}

    for row in rows:
        doc_id = row["doc_id"]
        label = row["label"]

        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "label": label,
                "n_chunks": 0,
            }

        else:
            #защита от ошибок данных
            if docs[doc_id]["label"] != label:
                raise ValueError(
                    f"Inconsistent label for doc_id: {doc_id}"
                )
        docs[doc_id]["n_chunks"] += 1

    return list(docs.values())

#%%
#стратификационный сплит по доументам
def stratified_doc_split(
        docs: List[Dict[str, Any]],
        test_size: float,
        val_size: float,
        random_state: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    doc_ids = [d["doc_id"] for d in docs]
    labels = [d["label"] for d in docs]

    train_val_ids, test_ids, train_val_y, _ = train_test_split(
        doc_ids,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    val_relative_size = val_size / (1.0 - test_size)

    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        train_val_y,
        test_size=val_relative_size,
        stratify=train_val_y,
        random_state=random_state,
    )

    doc_map = {d["doc_id"]: d for d in docs}

    train_docs = [doc_map[d] for d in train_ids]
    val_docs = [doc_map[d] for d in val_ids]
    test_docs = [doc_map[d] for d in test_ids]

    return train_docs, val_docs, test_docs

#%%
#добавляем сплит к чанкам
def add_split_to_chunks(
        rows: List[Dict[str, Any]],
        train_docs: List[Dict[str, Any]],
        val_docs: List[Dict[str, Any]],
        test_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    train_ids = {d["doc_id"] for d in train_docs}
    val_ids = {d["doc_id"] for d in val_docs}
    test_ids = {d["doc_id"] for d in test_docs}

    out: List[Dict[str, Any]] = []

    for row in rows:
        doc_id = row["doc_id"]

        if doc_id in train_ids:
            split = "train"

        elif doc_id in val_ids:
            split = "val"

        elif doc_id in test_ids:
            split = "test"

        else:
            raise ValueError("doc_id missing from split")

        new_row = dict(row)
        new_row["split"] = split

        out.append(new_row)

    return out

#%%
#статистика сплитов
def print_stats(name: str, docs: List[Dict[str, Any]]) -> None:
    label_counts = Counter(d["label"] for d in docs)
    chunk_count = sum(d["n_chunks"] for d in docs)

    print(f"\n=== {name.upper()} ===")
    print(f"docs: {len(docs)}")
    print(f"chunks: {chunk_count}")

    for label, count in sorted(label_counts.items()):
        print(f"label {label}: {count} docs")

#%%
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(DATASET_PATH)
    print(f"Loaded chunk rows: {len(rows)}")

    docs = build_doc_table(rows)
    print(f"Unique docs: {len(docs)}")

    train_docs, val_docs, test_docs = stratified_doc_split(
        docs=docs,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    print_stats("train", train_docs)
    print_stats("val", val_docs)
    print_stats("test", test_docs)

    rows_with_split = add_split_to_chunks(
        rows=rows,
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
    )

    split_manifest = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
        },
        "train_doc_ids": [d["doc_id"] for d in train_docs],
        "val_doc_ids": [d["doc_id"] for d in val_docs],
        "test_doc_ids": [d["doc_id"] for d in test_docs],
    }

    write_json(DOC_SPLITS_PATH, split_manifest)
    write_jsonl(CHUNK_SPLITS_PATH, rows_with_split)

    print("\nSaved:")
    print(f"- doc split manifest: {DOC_SPLITS_PATH}")
    print(f"- chunk dataset with split column: {CHUNK_SPLITS_PATH}")
#%%
if __name__ == "__main__":
    main()