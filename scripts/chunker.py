#%%
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

#сначала выделяем финальные знаки препинания "конца предложения"
_SENT_END_RE = re.compile(
    r"""
    #минимально адекватный сплит по концу предложения:
    # . ! ? … + возможные закрывающие кавычки/скобки после
    (?:
        [.!?…]+
        (?:["»')\]]+)?   #закрывающие символы после знака
    )
    \s+                 #пробел/перевод строки после
    """,
    re.VERBOSE
)

#для случаев, когда после конца текста нет пробела
_SENT_END_AT_EOF_RE = re.compile(r"[.!?…]+(?:[\"»')\]]+)?\s*$")

@dataclass
class Chunk:
    chunk_id: int
    text: str
    n_tokens: int
    sent_start: int #индекс первого предложения
    sent_end: int #индекс последнего предложения включительно

def whitespace_tokenize(s: str) -> list[str]:
    return [t for t in s.split() if t]

#быстрый счетчик длины для cnn baseline
def count_ws_tokens(s: str) -> int:
    return len(whitespace_tokenize(s))

_WORD_RE = re.compile(r"[а-яёa-z0-9]+|[.,!?;:()\"-]", flags=re.IGNORECASE)

def normalize_text_for_cnn(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_text_for_cnn(text: str) -> List[str]:
    text = normalize_text_for_cnn(text)
    return _WORD_RE.findall(text)

def count_cnn_tokens(text: str) -> int:
    return len(tokenize_text_for_cnn(text))

#грубая нарезка на предложения, возвращает список строк в исходном порядке
def split_into_sentences(text: str) -> List[str]:
    text = (text or "").strip()

    if not text:
        return []

    parts: List[str] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        end = m.end()
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)

    return parts

def make_sentence_chunks(
        text: str,
        max_tokens: int, #максимум длины чанка в единицах length_fn
        min_tokens: int, #минимальная длина, чанки короче будут склеиваться с соседом
        join_with: str=" ",
        length_fn: Callable[[str], int] = count_ws_tokens, #фнукция длины строки: CNN ws-tokens, BERT input_ids
        allow_merge_over_max: bool = True, #при скелйке коротких чанков можно слегка превысить max_length
) -> Tuple[List[Chunk], dict]:

    sents = split_into_sentences(text)

    if not sents:
        return [], {"empty": True, "n_sentences": 0, "n_chunks": 0, "n_tokens": 0}

    #собираем сырые чанки
    raw_texts: List[str] = []
    raw_spans: List[Tuple[int, int]] = []
    raw_lens: List[int] = []

    cur: List[str] = []
    cur_len = 0
    cur_start = 0
    too_long_sentences = 0

    for i, sent in enumerate(sents):
        sent_len = length_fn(sent)

        #если предложение > max_length, не дробим, кладем отдельным чанком
        if sent_len > max_tokens:
            too_long_sentences += 1

            if cur:
                t = join_with.join(cur).strip()
                raw_texts.append(t)
                raw_spans.append((cur_start, i - 1))
                raw_lens.append(cur_len)
                cur = []
                cur_len = 0

            raw_texts.append(sent.strip())
            raw_spans.append((i, i))
            raw_lens.append(sent_len)
            cur_start = i + 1
            continue

        if cur and (cur_len + sent_len > max_tokens):
            #закрываем текущий
            t = join_with.join(cur).strip()
            raw_texts.append(t)
            raw_spans.append((cur_start, i - 1))
            raw_lens.append(cur_len)

            #начинаем новый
            cur = [sent]
            cur_len = sent_len
            cur_start = i
        else:
            if not cur:
                cur_start = i
            cur.append(sent)
            cur_len += sent_len

    if cur:
        t = join_with.join(cur).strip()
        raw_texts.append(t)
        raw_spans.append((cur_start, len(sents) - 1))
        raw_lens.append(cur_len)

    #склейка коротких чанков
    merged_texts: List[str] = []
    merged_spans: List[Tuple[int, int]] = []
    merged_lens: List[int] = []

    i = 0
    while i < len(raw_texts):
        t = raw_texts[i]
        ln = raw_lens[i]
        st, en = raw_spans[i]

        #если окпо длине (или вообще один чанк) - переносим
        if ln >= min_tokens or len(raw_texts) == 1:
            merged_texts.append(t)
            merged_spans.append((st, en))
            merged_lens.append(ln)
            i += 1
            continue

        #короткий чанк: склеиваем со следущим, если он есть
        if i + 1< len(raw_texts):
            t2 = raw_texts[i + 1]
            ln2 = raw_lens[i + 1]
            st2, en2 = raw_spans[i + 1]

            merged_candidate = (t + join_with +t2).strip()

            #есть пара способов подсчета длины сейчас используется,
            #которая быстрее. второй варик: length_fn(merged_candidate
            merged_len = ln + ln2

            if allow_merge_over_max or merged_len <= max_tokens:
                merged_texts.append(merged_candidate)
                merged_spans.append((st, en2))
                merged_lens.append(merged_len)
                i += 2
            else:
                #если нельзя склеивать, оставляем как есть
                #чтобы не превысить max_tokens
                merged_texts.append(t)
                merged_spans.append((st, en))
                merged_lens.append(ln)
                i += 1
        else:
            #последний коротыш приклеивается к предыдущему, если он есть
            if merged_texts:
                merged_texts[-1] = (merged_texts[-1] + join_with + t).strip()
                prev_sent, prev_end = merged_spans[-1]
                merged_spans[-1] = (prev_sent, en)
                merged_lens[-1] += ln
            else:
                merged_texts.append(t)
                merged_spans.append((st, en))
                merged_lens.append(ln)
            i += 1

    chunks: List[Chunk] = []
    for cid, (t, ln, (st, en)) in enumerate(zip(merged_texts, merged_lens, merged_spans)):
        chunks.append(Chunk(chunk_id=cid, text=t, n_tokens=ln, sent_start=st, sent_end=en))

    diag = {
        "empty": False,
        "n_sents": len(sents),
        "n_chunks_raw": len(raw_texts),
        "n_chunks": len(chunks),
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "too_long_sentences": too_long_sentences,
        "allow_merge_over_max": allow_merge_over_max,
    }
    return chunks, diag

#%%
#короткий тест, можно удалить
if __name__ == "__main__":
    sample = (
        "Это первое предложение. "
        "Вот второе — чуть длиннее, но всё ещё ок! "
        "Третье? "
        "Четвёртое… "
        "А дальше идёт текст без точек который останется одним предложением если не встретит границы"
    )
    chunks, diag = make_sentence_chunks(sample, max_tokens=10, min_tokens=5)
    print(diag)
    for c in chunks:
        print("---", c.chunk_id, c.n_tokens)
        print(c.text)
