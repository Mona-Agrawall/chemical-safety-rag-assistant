"""Small dependency-free BM25 (Okapi) index for exact-term retrieval.

Dense embeddings are weak at matching identifiers such as CAS numbers,
H-statements, UN numbers and numeric limits; BM25 complements them.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./+][a-z0-9]+)*")
STOPWORDS = frozenset(
    (
        "a an and are as at be by can do does for from has have how i if in into is it its me of on or "
        "should than that the their there these this to was what when where which who why will with you "
        "your about any according give tell list please"
    ).split()
)
_ALNUM_SPLIT = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def _stem(tok: str) -> str:
    """Very light plural stemming so "rats" matches "rat", "limits" matches "limit"."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")) and tok.isalpha():
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    # PDF extraction often glues words together ("OralRat930", "numberUN"):
    # split at lower->Upper transitions before lowercasing.
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).lower()
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    tokens: list[str] = []
    for tok in TOKEN_RE.findall(text):
        tok = tok.rstrip(".")
        if not tok or tok in STOPWORDS:
            continue
        tokens.append(_stem(tok))
        if tok.replace("-", "").isdigit():
            continue  # identifiers such as CAS "67-64-1" stay whole only
        if "+" in tok:
            # Combined precautionary codes "p305+p351+p338" -> also each code.
            tokens.extend(p for p in tok.split("+") if p)
        elif "-" in tok:
            # "fire-fighting" -> also "fire", "fighting"
            tokens.extend(_stem(p) for p in tok.split("-") if p and p not in STOPWORDS and not p.isdigit())
        elif not tok.isalpha() and not tok.replace(".", "").isdigit():
            # Mixed letters/digits ("ld50", "un1114", "rat930"): also emit the parts.
            parts = _ALNUM_SPLIT.findall(tok)
            if len(parts) > 1:
                tokens.extend(_stem(p) for p in parts if p not in STOPWORDS and (len(p) > 1 or p.isdigit()))
    return tokens


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.idf = {
            term: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query: str, allowed: Iterable[int] | None = None) -> dict[int, float]:
        """BM25 score per matching document id (optionally restricted)."""
        allow = set(allowed) if allowed is not None else None
        out: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                if allow is not None and doc_id not in allow:
                    continue
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                out[doc_id] = out.get(doc_id, 0.0) + idf * tf * (self.k1 + 1) / denom
        return out

    def top(self, query: str, k: int, allowed: Iterable[int] | None = None) -> list[tuple[int, float]]:
        scored = self.scores(query, allowed)
        return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
