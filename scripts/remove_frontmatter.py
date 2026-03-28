#%%импорты и конфиги
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List

import pandas as pd
import logging

IN_JSONL = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/previews_children.jsonl")
OUT_JSONL = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/previews_children_nofront.jsonl")
OUT_REPORT = Path("/Volumes/Extreme SSD/vkr_rusage/inter_json/cleaning_report_children_nofront.csv")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("frontmatter")

#настраиваем эвристики
MAX_SCAN_LINES = 35 #сколько строк анализируем в начале
MAX_DROP_LINES = 60 #максимальный сред фронтматтера
MIN_NARRATIVE_LEN = 50 #строка похожа на нарратив, если достаточно длинная/пунктуация
TITLELIKE_MAX_WORDS = 6 #строка похожа на заголовк/автора, если очень короткая
TITLELIKE_THRESHOLD = 4 #сколько похожих на тайтл стро в первых MAX_SCAN_LINES
#%%эвристики
FRONT_CUES_RE = re.compile(
r"(isbn|©|все права защищены|издательств|литрес|правообладател"
    r"|аннотаци|серия|том\s+\d+|оглавлени|посвящает|эпиграф)",
    re.IGNORECASE
)

BAD_HEADER_RE = re.compile(
    r"(isbn|все права защищены|литрес|правообладател|издательств|©)",
    re.IGNORECASE
)

#на всякий случай проверим хвост, мало ли
TAIL_MARK_RE = re.compile(
    r"(конец ознакомительного фрагмента|ознакомительный фрагмент.*?закончен|"
    r"читайте.*?на литрес|купить.*?на литрес|вы можете купить|"
    r"the end of the sample)",
    re.IGNORECASE | re.DOTALL
)

#строка похожа на заголовок, если она короткая и не нарративная
def is_title_like_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return False

    #слишком много цифр/символов - часто служебное
    if sum(ch.isdigit() for ch in t) >=6:
        return True

    words = t.split()
    if len(words) <= TITLELIKE_MAX_WORDS:
        #если нет типичный нарративной пунктуации - верятно заголовок
        if not any(ch in t for ch in ".!?…,:;—–«»\""):
            return True

        #пример: Галина Врублевская - без точки, но с заглавных слов
        if all(w[:1].isupper() for w in words if w[:1].isalpha()) and not any(ch in t for ch in ".!?…"):
            return True
    return False

#строка похожа на начало художественного текста
def is_narrative_like_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if t == "<CHAPTER>":
        return True
    if len(t) >= MIN_NARRATIVE_LEN:
        return True
    #диалог/нарратив часто содержит пунктуацию
    if any(ch in t for ch in ".!?…«»\""):
        return True
    if t.startswith(("—", "–", "-")) and len(t) >= 15:
        return True
    return False

#на всякий случай, если хвост остался, отрезаем все от первого вхождения
def cut_tail_if_present(text: str) -> Tuple[str, bool]:
    m = TAIL_MARK_RE.search(text)
    if not m:
        return text, False
    return text[: m.start()].rstrip(), True

@dataclass
class FrontmatterDiag:
    removed_lines: int
    start_line_idx: int
    used_chapter_anchor: bool
    used_cues: bool
    titlelike_count: int
    bad_header_before: bool
    bad_header_after: bool
    tail_cut: bool
    orig_len: int
    new_len: int
#%%эвристическая чистка фронтматтера
def remove_frontmatter_smart(text: str) -> Tuple[str, FrontmatterDiag]:
    #диагностируем плохой хедер до
    before_lines = text.splitlines()
    bad_before = bool(BAD_HEADER_RE.search("\n".join(before_lines[:40])))

    text2, tail_cut = cut_tail_if_present(text)

    lines = text2.splitlines()
    scan = lines[:MAX_SCAN_LINES]

    used_chapter_anchor = False
    used_cues = False

    #если <CHAPTER> встречается рано, режем до него
    for i, ln in enumerate(scan):
        if ln.strip() == "<CHAPTER>":
            start_idx = i
            used_chapter_anchor = True
            break
    else:
        start_idx = 0

    #если нет якоря, решаем по признакам
    if not used_chapter_anchor:
        joined_scan = "\n".join(scan)
        if FRONT_CUES_RE.search(joined_scan):
            used_cues = True

        titlelike_count = sum(is_title_like_line(x) for x in scan if x.strip())
        has_frontmatter = used_cues or (titlelike_count >= TITLELIKE_THRESHOLD)

        if has_frontmatter:
            #ищем первую нарративную строку и стартуем от нее
            found = False
            for i, ln in enumerate(lines[:MAX_DROP_LINES]):
                if is_narrative_like_line(ln):
                    start_idx = i
                    found = True
                    break
            if not found:
                start_idx = 0
        else:
            titlelike_count = titlelike_count
    else:
        titlelike_count = sum(is_title_like_line(x) for x in scan if x.strip())

    #применяем срез
    new_lines = lines[start_idx:]

    #подчищаем ведущие пустые строки после среза
    while new_lines and not new_lines[0].strip():
        new_lines = new_lines[1:]
        start_idx += 1

    new_text = "\n".join(new_lines).strip()

    #диагностируем плохой хедер после
    after_lines = new_text.splitlines()
    bad_after = bool(BAD_HEADER_RE.search("\n".join(after_lines[:40])))

    diag = FrontmatterDiag(
        removed_lines=start_idx,
        start_line_idx=start_idx,
        used_chapter_anchor=used_chapter_anchor,
        used_cues=used_cues,
        titlelike_count=titlelike_count,
        bad_header_before=bad_before,
        bad_header_after=bad_after,
        tail_cut=tail_cut,
        orig_len=len(text.split()),
        new_len=len(new_text.split())
    )
    return new_text, diag
#%%трясемся и запускаемся
def main() -> None:
    assert IN_JSONL.exists(), f"Не найден входной файл: {IN_JSONL}"

    report_rows: List[Dict[str, Any]] = []
    n = 0

    with IN_JSONL.open("r", encoding="utf-8") as fin, \
        OUT_JSONL.open("w", encoding="utf-8") as fout:

        for line in fin:
            n += 1
            obj = json.loads(line)

            file_id = obj.get("id", "")
            text = obj.get("text", "")

            new_text, diag = remove_frontmatter_smart(text)

            obj_out = dict(obj)
            obj_out["text"] = new_text
            #сохраняю флаги прямо в записи на случай отладки
            obj_out.setdefault("meta", {})
            obj_out["meta"]["_frontmatter_removed_lines"] = diag.removed_lines
            obj_out["meta"]["_tail_cut"] = diag.tail_cut

            fout.write(json.dumps(obj_out, ensure_ascii=False) + "\n")

            report_rows.append({
                "file_id": file_id,
                "removed_lines": diag.removed_lines,
                "used_chapter_anchor": diag.used_chapter_anchor,
                "used_cues": diag.used_cues,
                "titlelike_count": diag.titlelike_count,
                "bad_header_before": diag.bad_header_before,
                "bad_header_after": diag.bad_header_after,
                "tail_cut": diag.tail_cut,
                "orig_len": diag.orig_len,
                "new_len": diag.new_len,
            })

            if n % 250 == 0:
                logger.info(f"Обработано {n}")

    df = pd.DataFrame(report_rows)
    df.to_csv(OUT_REPORT, index=False)
    logger.info(f"Готово. Записано: {n} строк")
    logger.info(f"JSONL: {OUT_JSONL}")
    logger.info(f"Report: {OUT_REPORT}")

    #сводка до/после
    if not df.empty:
        logger.info(f"bad_header_before mean: {df['bad_header_before'].mean():.3f}")
        logger.info(f"bad_header_after  mean: {df['bad_header_after'].mean():.3f}")
        logger.info(f"median orig_len_words: {df['orig_len'].median():.1f}")
        logger.info(f"median new_len_words : {df['new_len'].median():.1f}")
        logger.info(f"share used_chapter_anchor: {df['used_chapter_anchor'].mean():.3f}")
        logger.info(f"share removed_lines>0: {(df['removed_lines'] > 0).mean():.3f}")

if __name__ == "__main__":
    main()