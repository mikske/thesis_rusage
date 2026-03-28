# %%
from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import List, Dict, Any, Set

from torch.utils.data import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CURRENT_DIR.parent
sys.path.append(str(SCRIPTS_DIR))

from chunker import make_sentence_chunks, count_cnn_tokens

# %% paths & config
JSONL_PATH = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/previews_children_nofront.jsonl")

#для CNN пока такие параметры
CNN_MAX_TOKENS = 700
CNN_MIN_TOKENS = 120

#работаем с двумя возрастными группами
VALID_AGE_GROUPS: Set[int] = {1, 2}

#%%чтение jsonl файла целиком
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows

# %% делаем датасет
class CNNAgeDataset(Dataset):

    def __init__(
        self,
        jsonl_path: Path,
        max_tokens: int = CNN_MAX_TOKENS,
        min_tokens: int = CNN_MIN_TOKENS,
        valid_age_groups: Set[int] = VALID_AGE_GROUPS,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.valid_age_groups = valid_age_groups

        #здесь будет плоский список всех chunk-level samples
        self.samples: List[Dict[str, Any]] = []

        self._build()

    def _build(self) -> None:
        #читаем документ, фильтрум по меткам, превращаем в набр чанкам

        rows = read_jsonl(self.jsonl_path)

        for row in rows:
            meta = row.get("meta", {})
            text = row.get("text", "")
            doc_id = row.get("id")

            age_group_id = meta.get("age_group_id")

            #пропускаем все, что не относится к нужным классам
            if age_group_id not in self.valid_age_groups:
                continue

            #переводим исходные метки в бинарные label: 1 -> 0, 2 -> 1
            label = 0 if age_group_id == 1 else 1

            chunks, diag = make_sentence_chunks(
                text=text,
                max_tokens=self.max_tokens,
                min_tokens=self.min_tokens,
                length_fn=count_cnn_tokens,
            )

            for ch in chunks:
                self.samples.append(
                    {
                        "text": ch.text,
                        "label": label,
                        "age_group_id": age_group_id,
                        "doc_id": doc_id,
                        "chunk_id": ch.chunk_id,
                        "chunk_tokens": ch.n_tokens,
                        "source": "rusage",
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

# %% quick sanity check
if __name__ == "__main__":
    ds = CNNAgeDataset(JSONL_PATH)

    print("Total chunk samples:", len(ds))

    for i in range(3):
        item = ds[i]
        print(f"\n--- SAMPLE {i} ---")
        print("doc_id:", item["doc_id"])
        print("chunk_id:", item["chunk_id"])
        print("label:", item["label"])
        print("age_group_id:", item["age_group_id"])
        print("chunk_tokens:", item["chunk_tokens"])
        print("text preview:", item["text"][:300].replace("\n", " "))

# %% dataset stats
if __name__ == "__main__":
    ds = CNNAgeDataset(JSONL_PATH)

    label_counts = {0: 0, 1: 0}
    doc_ids = set()

    for item in ds.samples:
        label_counts[item["label"]] += 1
        doc_ids.add(item["doc_id"])

    print("\n=== DATASET STATS ===")
    print("unique docs:", len(doc_ids))
    print("chunk samples:", len(ds))
    print("label 0 (age_group_id=1):", label_counts[0])
    print("label 1 (age_group_id=2):", label_counts[1])

    OUTPUT_PATH = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/cnn_chunks_dataset.jsonl")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in ds.samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nDataset saved to: {OUTPUT_PATH}")
    print(f"Total samples written: {len(ds.samples)}")