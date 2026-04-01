from __future__ import annotations

from pathlib import Path
import json
import argparse
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

#%%
#строим doc_split один раз, далее применяем его к датасетам
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--splits_path", type=str, required=True)

    #1 --- строим новый split и применяем его
    #0 --- только применяем уже готовый split
    parser.add_argument("--build_splits", type=int, default=1)

    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)

    return parser.parse_args()

#%% читаем jsonl
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

#%%
#запись jsonl (для chunk dataset)
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

#%%
#чтение json
def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

#%%
#запись json (для split manifest)
def write_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

#%%
#строим таблицу на уровне документов из датасетов уровня чанков
#одна строка --- один документ
def build_doc_table(chunk_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not chunk_rows:
        raise ValueError("Пустой chunk dataset.")

    df = pd.DataFrame(chunk_rows)

    required_cols = {"doc_id", "label", "chunk_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В chunk dataset не хватает колонок: {missing}")

    #у одного документа должна быть только одна метка
    label_nunique = df.groupby("doc_id")["label"].nunique()
    bad_docs = label_nunique[label_nunique > 1]

    if len(bad_docs) > 0:
        raise ValueError(
            f"Найдены документы с несколькими label: {len(bad_docs)}. "
            f"Примеры doc_id: {bad_docs.index.tolist()[:10]}"
        )

    doc_df = (
        df.groupby("doc_id", as_index=False)
        .agg(
            label=("label", "first"),
            n_chunks=("chunk_id", "count"),
        )
        .copy()
    )

    doc_df = doc_df.sort_values("doc_id").reset_index(drop=True)
    return doc_df

#%%
#делаем stratified split по документам
def make_doc_split(
    doc_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, List[str]]:

    required_cols = {"doc_id", "label"}
    missing = required_cols - set(doc_df.columns)
    if missing:
        raise ValueError(f"В doc_df не хватает колонок: {missing}")

    train_val_docs, test_docs = train_test_split(
        doc_df,
        test_size=test_size,
        random_state=random_state,
        stratify=doc_df["label"],
    )

    val_relative_size = val_size / (1.0 - test_size)

    train_docs, val_docs = train_test_split(
        train_val_docs,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_docs["label"],
    )

    split_manifest = {
        "train_doc_ids": sorted(train_docs["doc_id"].tolist()),
        "val_doc_ids": sorted(val_docs["doc_id"].tolist()),
        "test_doc_ids": sorted(test_docs["doc_id"].tolist()),
        "config": {
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
        },
    }

    return split_manifest

#%%
#добавляем сплиты к чанкам
def add_split_to_chunks(
    chunk_rows: List[Dict[str, Any]],
    split_manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    train_ids = set(split_manifest["train_doc_ids"])
    val_ids = set(split_manifest["val_doc_ids"])
    test_ids = set(split_manifest["test_doc_ids"])

    out_rows: List[Dict[str, Any]] = []
    missing_doc_ids = set()

    for row in chunk_rows:
        doc_id = row.get("doc_id")

        if doc_id in train_ids:
            split = "train"
        elif doc_id in val_ids:
            split = "val"
        elif doc_id in test_ids:
            split = "test"
        else:
            missing_doc_ids.add(doc_id)
            continue

        row = row.copy()
        row["split"] = split
        out_rows.append(row)

    if missing_doc_ids:
        raise ValueError(
            f"Не нашли split для {len(missing_doc_ids)} doc_id. "
            f"Примеры: {list(missing_doc_ids)[:10]}"
        )

    return out_rows

#%%
#выводим статистику
def print_stats(doc_df: pd.DataFrame, split_manifest: Dict[str, Any]) -> None:
    train_ids = set(split_manifest["train_doc_ids"])
    val_ids = set(split_manifest["val_doc_ids"])
    test_ids = set(split_manifest["test_doc_ids"])

    print("\n=== DOC DATASET STATS ===")
    print(f"unique docs: {len(doc_df)}")

    print("\n=== FULL LABEL DISTRIBUTION ===")
    print(doc_df["label"].value_counts().sort_index())

    #делим doc_df по split
    train_df = doc_df[doc_df["doc_id"].isin(train_ids)].copy()
    val_df = doc_df[doc_df["doc_id"].isin(val_ids)].copy()
    test_df = doc_df[doc_df["doc_id"].isin(test_ids)].copy()

    print("\n=== SPLIT COUNTS ===")
    print(f"train docs: {len(train_df)}")
    print(f"val docs: {len(val_df)}")
    print(f"test docs: {len(test_df)}")

    print("\n=== LABEL DISTRIBUTION BY SPLIT ===")

    print("\ntrain:")
    print(train_df["label"].value_counts().sort_index())

    print("\nval:")
    print(val_df["label"].value_counts().sort_index())

    print("\ntest:")
    print(test_df["label"].value_counts().sort_index())

    #нормализованные доли
    print("\n=== NORMALIZED (ROW-WISE) ===")

    def normalize_counts(df: pd.DataFrame) -> pd.Series:
        counts = df["label"].value_counts(normalize=True).sort_index()
        return counts.round(4)

    print("\ntrain:")
    print(normalize_counts(train_df))

    print("\nval:")
    print(normalize_counts(val_df))

    print("\ntest:")
    print(normalize_counts(test_df))

#%%
def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    splits_path = Path(args.splits_path)

    chunk_rows = read_jsonl(input_path)
    print(f"[INFO] loaded chunk rows: {len(chunk_rows)}")

    if args.build_splits == 1:
        #строим split с нуля
        doc_df = build_doc_table(chunk_rows)

        split_manifest = make_doc_split(
            doc_df=doc_df,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )

        print_stats(doc_df, split_manifest)
        write_json(splits_path, split_manifest)
        print(f"[SAVED] doc splits: {splits_path}")

    else:
        #используем уже готовый split
        split_manifest = read_json(splits_path)
        print(f"[INFO] loaded existing doc splits: {splits_path}")

    chunk_rows_with_split = add_split_to_chunks(chunk_rows, split_manifest)
    write_jsonl(output_path, chunk_rows_with_split)

    print(f"[SAVED] chunks with split: {output_path}")
    print(f"[INFO] total chunk rows written: {len(chunk_rows_with_split)}")

#%%
if __name__ == "__main__":
    main()