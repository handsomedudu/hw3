from __future__ import annotations

import re
from typing import Callable


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b")
_NUM_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[^a-zA-Z\s]")


def basic_text_clean(text: str) -> str:
    """Basic text cleanup: lower, strip URLs/emails/numbers/punct, collapse spaces."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _NUM_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def make_sklearn_preprocessor() -> Callable[[str], str]:
    """Returns a function usable as scikit-learn's TfidfVectorizer(preprocessor=...)."""
    return basic_text_clean

