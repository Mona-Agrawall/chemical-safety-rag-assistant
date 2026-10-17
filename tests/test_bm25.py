import pytest

from rag.bm25 import BM25, tokenize


@pytest.mark.parametrize(
    ("text", "expected_subset"),
    [
        ("CAS number:71-43-2", {"cas", "number", "71-43-2"}),
        ("LD50 OralRat930 mg/kg", {"ld50", "oral", "rat", "930"}),
        ("UN1114UN1114DOT", {"un", "1114"}),
        ("Lower explosion limit 2,6 vol%", {"2.6", "limit", "explosion"}),
        ("P305+P351+P338 IF IN EYES", {"p305+p351+p338", "p305", "p351", "eye"}),
        ("fire-fighting measures", {"fire-fighting", "fire", "fighting", "measure"}),
        ("Flash point −17 °C", {"flash", "point", "17"}),
    ],
)
def test_tokenize(text, expected_subset):
    assert expected_subset <= set(tokenize(text))


def test_stopwords_removed():
    assert tokenize("What is the flash point of it?") == ["flash", "point"]


def test_cas_numbers_are_not_split():
    toks = tokenize("67-64-1")
    assert toks == ["67-64-1"]


def test_bm25_ranks_exact_identifier_first():
    docs = [
        "Acetone CAS number 67-64-1 identification",
        "Methanol CAS number 67-56-1 identification",
        "Ethanol flash point 12 C",
    ]
    bm = BM25(docs)
    assert bm.top("CAS 67-56-1", 1)[0][0] == 1
    assert bm.top("flash point", 1)[0][0] == 2


def test_bm25_allowed_filter_and_unknown_terms():
    bm = BM25(["alpha beta", "alpha gamma", "delta"])
    assert {i for i, _ in bm.top("alpha", 5, allowed=[1, 2])} == {1}
    assert bm.top("zzz", 5) == []


def test_bm25_empty_corpus():
    assert BM25([]).top("anything", 3) == []
