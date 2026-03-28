# %%
from __future__ import annotations
from pathlib import Path

import json
from pathlib import Path
from statistics import mean, median

from chunker import make_sentence_chunks

JSONL_PATH = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/previews_children_nofront.jsonl")
SUMMARY_OUT = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/chunking_summary_cnn.txt")

MAX_DOCS = 20
MAX_TOKENS = 700
MIN_TOKENS = 120


def main():
    all_chunk_counts = []
    all_chunk_lengths = []

    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_DOCS:
                break

            obj = json.loads(line)
            text = obj["text"]
            doc_id = obj["id"]

            chunks, diag = make_sentence_chunks(
                text=text,
                max_tokens=MAX_TOKENS,
                min_tokens=MIN_TOKENS,
            )

            all_chunk_counts.append(len(chunks))
            all_chunk_lengths.extend([c.n_tokens for c in chunks])

            print(f"\n=== DOC {i+1}: {doc_id} ===")
            print(diag)
            print(f"chunks: {len(chunks)}")

            for c in chunks[:2]:
                print(f"--- chunk {c.chunk_id} | tokens={c.n_tokens}")
                print(c.text[:400].replace("\n", " "))
                print()

    print("\n=== SUMMARY ===")
    print(f"docs processed: {len(all_chunk_counts)}")
    print(f"avg chunks per doc: {mean(all_chunk_counts):.2f}")
    print(f"median chunks per doc: {median(all_chunk_counts):.2f}")
    print(f"avg chunk length: {mean(all_chunk_lengths):.2f}")
    print(f"median chunk length: {median(all_chunk_lengths):.2f}")
    print(f"min chunk length: {min(all_chunk_lengths)}")
    print(f"max chunk length: {max(all_chunk_lengths)}")

    summary_lines = [
        "=== SUMMARY ===",
        f"docs processed: {len(all_chunk_counts)}",
        f"avg chunks per doc: {mean(all_chunk_counts):.2f}",
        f"median chunks per doc: {median(all_chunk_counts):.2f}",
        f"avg chunk length: {mean(all_chunk_lengths):.2f}",
        f"median chunk length: {median(all_chunk_lengths):.2f}",
        f"min chunk length: {min(all_chunk_lengths)}",
        f"max chunk length: {max(all_chunk_lengths)}",
    ]

    summary_text = "\n".join(summary_lines)

    summary_out = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/chunking_summary_cnn.txt")
    summary_out.write_text(summary_text, encoding="utf-8")

    print(f"\nSummary saved to: {summary_out}")

if __name__ == "__main__":
    main()
