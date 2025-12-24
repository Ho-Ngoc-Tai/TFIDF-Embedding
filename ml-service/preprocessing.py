from __future__ import annotations

import re
from typing import Iterable, List

from underthesea import word_tokenize

VIETNAMESE_STOPWORDS = {
    "và", "của", "là", "có", "được", "cho", "với", "các", "này", "trong",
    "để", "những", "một", "người", "không", "đã", "khi", "từ", "như", "còn",
    "theo", "đến", "về", "tại", "cũng", "nhưng", "hay", "đó", "vì", "nếu",
    "nhiều", "làm", "ra", "nên", "thì", "đang", "sẽ", "vào", "lên", "bị",
    "sau", "trước", "rất", "đây", "chỉ", "hơn", "bởi", "mà", "qua", "lại",
    "thế", "năm", "ngày", "trên", "dưới", "việc", "đều", "tôi", "anh", "chị",
    "em", "họ", "chúng", "ta", "mình", "nào", "gì", "ai", "đâu", "sao",
    "vậy", "ấy", "kia", "đấy", "thôi", "nhé", "ạ", "ơi", "à", "ừ",
}

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_NON_WORD_PATTERN = re.compile(r"[^a-zA-ZÀ-ỹà-ỹ\s]")
_WS_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _NON_WORD_PATTERN.sub(" ", text)
    text = _WS_PATTERN.sub(" ", text)
    return text.strip()


def tokenize_vietnamese(text: str, stopwords: Iterable[str] | None = None) -> List[str]:
    stopwords = set(stopwords) if stopwords is not None else VIETNAMESE_STOPWORDS
    tokenized_text = word_tokenize(text, format="text")
    tokens = [
        token
        for token in tokenized_text.split()
        if token not in stopwords and len(token) > 1
    ]
    return tokens
