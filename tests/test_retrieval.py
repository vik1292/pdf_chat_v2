from retrieval import HybridIndex, rrf_fuse, select_evidence, tokenize
from tests.fakes import FakeEmbedder


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Tuition-Reimbursement, up to $5,000!") == [
        "tuition",
        "reimbursement",
        "up",
        "to",
        "5",
        "000",
    ]


def test_rrf_fuse_rewards_documents_ranked_by_both_lists():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == "a"  # ranks 1 and 2
    assert ids[1] == "c"  # ranks 3 and 1
    assert set(ids) == {"a", "b", "c", "d"}
    assert fused[0][1] == 1 / 61 + 1 / 62


def test_rrf_fuse_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_hybrid_index_finds_keyword_and_semantic_matches():
    index = HybridIndex(FakeEmbedder())
    index.add(
        ["c1", "c2", "c3"],
        [
            "The company offers a tuition reimbursement program.",
            "Full time employees accrue paid time off.",
            "Hard hats are required on the factory floor.",
        ],
    )
    assert len(index) == 3
    results = index.search("tuition reimbursement", k=3)
    assert results[0][0] == "c1"
    assert all(isinstance(score, float) for _, score in results)
    index.close()


def test_hybrid_index_search_on_empty_index_returns_nothing():
    index = HybridIndex(FakeEmbedder())
    assert index.search("anything") == []
    index.close()


def test_hybrid_index_k_larger_than_corpus_is_safe():
    index = HybridIndex(FakeEmbedder())
    index.add(["c1"], ["only one chunk here"])
    assert [i for i, _ in index.search("chunk", k=50)] == ["c1"]
    index.close()


def test_select_evidence_keeps_top_and_near_top_scores():
    # best is 8.0; margin 2.0 keeps >= 6.0; max_items caps at 3
    assert select_evidence([1.0, 8.0, 7.5, 6.0, 5.9], margin=2.0, max_items=3) == [1, 2, 3]


def test_select_evidence_always_returns_at_least_the_best():
    assert select_evidence([-9.0, -3.0, -8.0]) == [1]
    assert select_evidence([]) == []
