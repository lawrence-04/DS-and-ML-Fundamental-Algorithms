import math

import pytest

from tf_idf import (
    compute_inverse_document_frequency,
    compute_term_frequency,
    compute_tf_idf_vector,
    split_document_into_words,
    vectorise_corpus,
)


def test_split_document_into_words():
    doc = "hello world hello"
    expected = ["hello", "world", "hello"]
    assert split_document_into_words(doc) == expected


def test_compute_term_frequency():
    words = ["hello", "world", "hello"]
    result = compute_term_frequency(words)
    assert pytest.approx(result["hello"], 0.01) == 2 / 3
    assert pytest.approx(result["world"], 0.01) == 1 / 3


def test_compute_inverse_document_frequency():
    split_docs = [["hello", "world"], ["hello", "machine"], ["goodbye", "world"]]
    idf = compute_inverse_document_frequency(split_docs)
    assert pytest.approx(idf["hello"], 0.01) == math.log(3 / 2)
    assert pytest.approx(idf["world"], 0.01) == math.log(3 / 2)
    assert pytest.approx(idf["machine"], 0.01) == math.log(3 / 1)
    assert pytest.approx(idf["goodbye"], 0.01) == math.log(3 / 1)


def test_compute_tf_idf_vector():
    tf = {"hello": 0.5, "world": 0.5}
    idf = {"hello": math.log(2), "world": math.log(3)}
    tfidf = compute_tf_idf_vector(tf, idf)
    assert pytest.approx(tfidf["hello"], 0.01) == 0.5 * math.log(2)
    assert pytest.approx(tfidf["world"], 0.01) == 0.5 * math.log(3)


def test_vectorise_corpus():
    corpus = ["hello world", "hello machine", "goodbye world"]
    tfidf_vectors = vectorise_corpus(corpus)
    assert len(tfidf_vectors) == 3
    assert "hello" in tfidf_vectors[0]
    assert "world" in tfidf_vectors[0]
