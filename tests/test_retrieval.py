import pytest

from rag.retrieval import Retriever, ScopeReason, rrf_fuse


@pytest.fixture
def retriever(kb, corpus, embedder):
    return Retriever(kb, embedder, corpus)


def chems(result):
    return {r.chunk.chemical_key for r in result.chunks}


def test_rrf_fuse_rewards_agreement():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert max(fused, key=fused.get) == 1
    assert set(fused) == {1, 2, 3, 4}


def test_single_chemical_question_only_returns_that_chemical(retriever):
    res = retriever.retrieve("What is the flash point of methanol?")
    assert res.scope == ("methanol",) and res.scope_reason is ScopeReason.QUESTION
    assert chems(res) == {"methanol"}
    assert any("9.7 C" in r.chunk.text for r in res.chunks)


def test_without_filter_similar_documents_contaminate(kb, corpus, embedder):
    """Ablation: documents share wording, so unfiltered search mixes chemicals."""
    unfiltered = Retriever(kb, embedder, corpus, chemical_filtering=False)
    res = unfiltered.retrieve("What is the flash point of methanol?", top_k=6)
    assert len(chems(res)) > 1


def test_alias_and_cas_scope(retriever):
    assert retriever.retrieve("flash point of methyl alcohol").scope == ("methanol",)
    assert retriever.retrieve("which substance is 64-17-5?").scope == ("ethanol",)


def test_multi_chemical_question_gets_evidence_for_each(retriever):
    res = retriever.retrieve("Compare the flash points of acetone and methanol", top_k=4)
    assert res.scope == ("acetone", "methanol")
    keys = [r.chunk.chemical_key for r in res.chunks]
    assert keys.count("acetone") >= 2 and keys.count("methanol") >= 2
    assert keys[0] != keys[1], "results are interleaved per chemical"


def test_question_about_unindexed_chemical_returns_nothing(retriever):
    res = retriever.retrieve("What is the flash point of toluene?")
    assert res.chunks == [] and res.mention.missing == ("toluene",)


def test_follow_up_inherits_previous_chemical(retriever):
    res = retriever.retrieve("And what is its UN number?", history_chemicals=("ethanol",))
    assert res.scope == ("ethanol",) and res.scope_reason is ScopeReason.HISTORY
    assert chems(res) == {"ethanol"}


def test_explicit_chemical_overrides_history(retriever):
    res = retriever.retrieve("What about acetone?", history_chemicals=("ethanol",))
    assert res.scope == ("acetone",)


def test_user_scope_override(retriever):
    res = retriever.retrieve("What gloves should I wear?", scope_override=("acetone",))
    assert res.scope_reason is ScopeReason.USER and chems(res) == {"acetone"}
    res = retriever.retrieve("flash point of methanol", scope_override=("acetone",))
    assert chems(res) == {"acetone"} and res.notes


def test_unscoped_question_searches_everything(retriever):
    res = retriever.retrieve("Which extinguishing media are unsuitable?", top_k=6)
    assert res.scope == () and res.scope_reason is ScopeReason.NONE and len(chems(res)) > 1


def test_identifier_found_by_keyword_search(kb, corpus, embedder):
    hybrid = Retriever(kb, embedder, corpus)
    res = hybrid.retrieve("UN 1230", top_k=1)
    assert "1230" in res.chunks[0].chunk.text


def test_reranker_reorders_and_failures_fall_back(kb, corpus, embedder):
    class Prefer9:
        def score(self, query, passages):
            return [1.0 if "9.7" in p else 0.0 for p in passages]

    res = Retriever(kb, embedder, corpus, Prefer9()).retrieve("methanol hazards", top_k=2)
    assert "9.7" in res.chunks[0].chunk.text

    class Broken:
        def score(self, query, passages):
            raise RuntimeError("model crashed")

    res = Retriever(kb, embedder, corpus, Broken()).retrieve("methanol hazards", top_k=2)
    assert len(res.chunks) == 2


def test_empty_knowledge_base(settings, embedder):
    from rag.indexing import load_knowledge_base, sync_knowledge_base

    sync_knowledge_base(settings, embedder)  # no documents at all
    kb = load_knowledge_base(settings)
    res = Retriever(kb, embedder, settings).retrieve("anything")
    assert res.chunks == [] and res.notes
