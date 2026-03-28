#%% импорты и конфиги
from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
#%% пути
TEXTS_DIR = Path("/Volumes/Extreme SSD/vkr_rusage/selected_previews")
OUT_DIR = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json")
METADATA_CSV = Path('/Volumes/Extreme SSD/vkr_rusage/full_metadata.csv')

OUT_JSONL = OUT_DIR / "previews_cleaned.jsonl"
OUT_REPORT = OUT_DIR / "cleaning_report.csv"

STRUCT_TOKEN_CHAPTER = "<CHAPTER>"
#%% простой логгер для отслеживания важных событий
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("json_creation")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
#%% подгрузка метаданных
ID_COLUMN = "file_id"

def load_metadata(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(csv_path)

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"Нет колонки '{ID_COLUMN}",
            f"Есть колонки '{list(df.columns)}'",
        )

    #на всякий случай приводим к строке, чтобы совпадало с .stem
    df[ID_COLUMN] = (
        df[ID_COLUMN]
        .astype(str)
        .str.replace("\ufeff", "", regex=False)  # BOM
        .str.strip()  # пробелы по краям
        .str.replace(".txt", "", regex=False)
        .str.strip()
    )

    #делаем мапу
    meta_map = {
        row[ID_COLUMN]: row.to_dict()
        for _, row in df.iterrows()
    }
    return meta_map
#%%
logger.info(f"Читаю метаданные: {METADATA_CSV}")
logger.info(f"TXT dir: {TEXTS_DIR}")
#%%
metadata_map = load_metadata(METADATA_CSV)
logger.info(f"Метаданные загружены: {len(metadata_map)} записей (ключ: {ID_COLUMN})")
#%% прописываем правила очистки документов с текстами
#отрезаем хвост Литрес как только находим маркер обрезаем все, что ниже
TAIL_PATTERNS = [
    r"Конец ознакомительного фрагмента",
    r"Текст предоставлен ООО\s+«?ЛитРес»?",
    r"Прочитайте эту книгу целиком",
    r"ЛитРес[:\.]" #добавляю на всякий случай
]

TAIL_RE = re.compile("|".join(f"(?:{p})" for p in TAIL_PATTERNS), re.IGNORECASE)

#также обозначаем витринные признаки
BAD_HEADER_RE = re.compile(
r"(isbn|все права защищены|литрес|правообладател|издательств|©|all rights reserved)",
    re.IGNORECASE
)

#и маркеры глав/частей/пр., их нормализуем в <CHAPTER>
CHAPTER_CUE_RE = re.compile(
    r"^\s*(глава\s+\d+|глава\s+[ivxlcdm]+|пролог|эпилог|част[ья]\s+\S+|\d+|[IVXLCDM]+|\*+\s*\*+\s*\*+)\s*$",
    re.IGNORECASE
)
#%% здесь прописываем эвристику
#отрезаем хвосты
def cut_litres_tails(text:str) -> Tuple[str, bool]:
    m = TAIL_RE.search(text)
    if m:
        return text[:m.start()], True
    return text, False

#делаем грубую проверку "похоже ли на худ. текст" (есть кириллица, пунктуация/диалоги, нет служебных маркеров)
def is_narrativeish(line:str) -> bool:
    t = line.strip()
    if not t:
        return False
    if BAD_HEADER_RE.search(t):
        return False
    has_cyr = bool(re.search(r"[А-Яа-яЁё]", t))
    has_dialogue = ("—" in t) or ("–" in t)
    has_punct = any(ch in t for ch in "«»…,.!?:;")
    return has_cyr and (has_dialogue or has_punct or len(t) >= 40)

#если строка маркер главы, возвращаем <CHAPTER>, иначе NONE
def normalize_structure_line(line:str) -> str:
    if CHAPTER_CUE_RE.match(line.strip()):
        return STRUCT_TOKEN_CHAPTER
    return None

#возвращаем индекс строки, с которой начинается основной текст
#логика: ищем сильный сигнал старта (глава/пролог/1), проверяем, что дальше достаточно narrative-ish строк
#логика: fallback: ищем первый “устойчивый” участок narrative-ish строк подряд
def find_main_start(lines: list[str], max_scan: int = 900) -> int:
    n = min(max_scan, len(lines))

    #ищем маркер и подтверждение окна
    for i in range(n):
        ln = lines[i].strip()
        if not ln:
            continue

        if CHAPTER_CUE_RE.match(ln):
            window = [x for x in lines[i+1:i+25] if x.strip()]
            score = sum(is_narrativeish(x) for x in window)
            if score >= 6:
                return i

    #fallback
    run = 0
    for i in range(n):
        if is_narrativeish(lines[i]):
            run += 1
            if run >= 7:
                return max(0, i-6)

        else:
            run = 0

    return 0
#%%чистим
def clean_preview_text(raw: str) -> tuple[str, dict]:
    raw = (
        raw
        .replace("\xa0", " ")  # NBSP
        .replace("\u2009", " ")  # thin space
        .replace("\u202f", " ")  # narrow NBSP
    )

    diag: dict = {}

    raw = raw.replace("\r\n", "\n")

    #отрезаем хвост
    trimmed, was_cut = cut_litres_tails(raw)
    diag["tail_cut"] = was_cut

    #разбиваем на строки
    lines = [ln.rstrip() for ln in trimmed.split("\n")]

    #убираем пустые строки вначале
    while lines and not lines[0].strip():
        lines.pop(0)

    #находим старт
    start_idx = find_main_start(lines)
    diag["start_idx"] = start_idx
    lines = lines[start_idx:]

    #нормализуем и чистим пробелы
    out_lines: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out_lines.append("")
            continue

        norm = normalize_structure_line(s)
        if norm is not None:
            if not out_lines or out_lines[-1] != STRUCT_TOKEN_CHAPTER:
                out_lines.append(STRUCT_TOKEN_CHAPTER)
            continue

        out_lines.append(s)

    text = "\n".join(out_lines)

    #схлопываем множественные пустые строки
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    #диагностика качества
    diag["len_chars"] = len(text)
    diag["len_words"] = len(re.findall(r"\S+", text))
    diag["bad_header_in_first_40_lines"] = bool(BAD_HEADER_RE.search("\n".join(out_lines[:40])))

    return text, diag
#%% основной прогон (тест)
OUT_DIR.mkdir(parents=True, exist_ok=True)

report_rows: list[dict] = []

DRY_RUN =  False
MAX_FILES = 20 #использовалось для теста

txt_files = sorted(TEXTS_DIR.glob("*.txt"))
logger.info(f"Найдено txt файлов: {len(txt_files)}")

written = 0
missing_meta = 0

with OUT_JSONL.open("w", encoding="utf-8") as out:
    for idx, text_path in enumerate(txt_files):
        if DRY_RUN and idx >= MAX_FILES:
            break

        file_id = text_path.stem #имя файла без txt

        meta =metadata_map.get(file_id)
        meta_found = meta is not None
        if not meta_found:
            missing_meta += 1
            meta = {"_meta_missing": True}

        raw = text_path.read_text(encoding="utf-8", errors="replace")
        text, diag = clean_preview_text(raw)

        record = {
            "id": file_id,
            "meta": meta, #все колонки CSV
            "text": text,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1

        report_rows.append({
            "file_id": file_id,
            "path": str(text_path),
            "meta_found": meta_found,
            **diag
        })

        if written % 200 == 0:
            logger.info(f"Обработано: {written}/len(txt_files)")

logger.info(f"Записано в JSONL: {written} строк {OUT_JSONL}")
logger.info(f"Файлов без метаданных: {missing_meta}")
#%%сохраняем отчет

df_report = pd.DataFrame(report_rows)
df_report.to_csv(OUT_REPORT, index=False)
logger.info(f"Отчёт сохранён: {OUT_REPORT}")

#мини-статистика
if not df_report.empty:
    logger.info(f"bad_head_in_first_40_lines: {df_report['bad_header_in_first_40_lines'].mean():.3f}")
    logger.info(f"median len_words: {df_report['len_words'].median()}")
# %% create children-only jsonl
IN_JSONL = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/previews_cleaned.jsonl")
OUT_CHILDREN_JSONL = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/previews_children.jsonl")

n_all = 0
n_children = 0

with IN_JSONL.open("r", encoding="utf-8") as fin, \
     OUT_CHILDREN_JSONL.open("w", encoding="utf-8") as fout:

    for line in fin:
        n_all += 1
        obj = json.loads(line)

        # надёжнее смотреть meta.file_id
        file_id = str(obj.get("meta", {}).get("file_id", obj.get("id", "")))
        file_id = file_id.replace(".txt", "").strip()

        if file_id.startswith("children_"):
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_children += 1

print(f"Total in original JSONL: {n_all}")
print(f"Children-only records: {n_children}")
print(f"Saved to: {OUT_CHILDREN_JSONL}")
#%%отчет по новому jsonl
CHILDREN_JSONL = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/previews_children.jsonl")
OUT_CHILDREN_REPORT = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/cleaning_report_children.csv")

BAD_HEAD_RE = re.compile(
    r"(isbn|все права защищены|литрес|правообладател|издательств|©)",
    re.IGNORECASE
)

rows = []

with CHILDREN_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj["text"]
        file_id = obj["id"]

        lines = text.splitlines()

        rows.append({
            "file_id": file_id,
            "len_chars": len(text),
            "len_whitespace_tokens": len(text.split()),
            "bad_header_in_first_40_lines": bool(
                BAD_HEAD_RE.search("\n".join(lines[:40]))
            )
        })

df_children_report = pd.DataFrame(rows)
df_children_report.to_csv(OUT_CHILDREN_REPORT, index=False)

print("Children-only report saved to:", OUT_CHILDREN_REPORT)
print("Median len_whitespace_tokens:", df_children_report["len_whitespace_tokens"].median())
print(
    "bad_header_in_first_40_lines:",
    df_children_report["bad_header_in_first_40_lines"].mean()
)
#%%age distribution for children-only jsonl
"""
Смотрим распределение возрастов в children-only корпусе.

Берём данные ТОЛЬКО из previews_children.jsonl,
чтобы статистика соответствовала реальному корпусу,
который пойдёт в классификацию/генерацию.
"""

import json
import pandas as pd
from pathlib import Path

CHILDREN_JSONL = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/previews_children.jsonl")

rows = []

with CHILDREN_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        meta = obj.get("meta", {})

        rows.append({
            "file_id": obj.get("id"),
            "age": meta.get("age"),
            "age_group_label": meta.get("age_group_label"),
            "age_group_id": meta.get("age_group_id"),
        })

df = pd.DataFrame(rows)

print("Total children texts:", len(df))
print("\nAge distribution:")
print(df["age"].value_counts().sort_index())

print("\nAge group label distribution:")
print(df["age_group_label"].value_counts())

print("\nAge group id distribution:")
print(df["age_group_id"].value_counts().sort_index())
