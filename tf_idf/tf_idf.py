import math
from collections import Counter, defaultdict
from functools import reduce


def split_document_into_words(document: str) -> list[str]:
    words = document.split(" ")
    words = [w.lower().strip() for w in words]
    return words


def compute_term_frequency(split_document: list[str]) -> dict[str, float]:
    term_counts = Counter(split_document)

    term_sum = sum(term_counts.values())
    assert term_sum != 0, "Tried to compute the term frequency on an empty set."

    document_frequency = {term: count / term_sum for term, count in term_counts.items()}
    return document_frequency


def compute_inverse_document_frequency(
    split_documents: list[list[str]],
) -> dict[str, float]:
    number_of_documents = len(split_documents)

    # get all terms in the corpus
    document_term_sets = [set(terms) for terms in split_documents]
    all_terms = reduce(lambda set_1, set_2: set_1.union(set_2), document_term_sets)

    inverse_document_frequency = defaultdict(float)
    for term in all_terms:
        term_document_count = 0
        for document in document_term_sets:
            if term in document:
                term_document_count += 1

        inverse_document_frequency[term] = math.log(
            number_of_documents / term_document_count
        )

    return inverse_document_frequency


def compute_tf_idf_vector(
    term_frequencies: dict[str, float], inverse_document_frequency: dict[str, float]
) -> dict[str, float]:
    tf_idf_vector = defaultdict(float)
    for term, frequency in term_frequencies.items():
        tf_idf_vector[term] = frequency * inverse_document_frequency[term]

    return tf_idf_vector


def vectorise_corpus(corpus: list[str]) -> list[dict[str, float]]:
    # preprocess the documents into terms
    split_documents = [split_document_into_words(document) for document in corpus]

    # compute the term-frequency
    document_term_frequencies = [
        compute_term_frequency(terms) for terms in split_documents
    ]

    # compute the inverse-document frequency
    inverse_document_frequency = compute_inverse_document_frequency(split_documents)

    # combine these into the final vectors
    tf_idf_vectors = [
        compute_tf_idf_vector(term_frequencies, inverse_document_frequency)
        for term_frequencies in document_term_frequencies
    ]

    return tf_idf_vectors
