# TF-IDF
Term Frequency-Inverse Document Frequency is a method for evaluating the importance of a term in a document, from a collection of documents.

It is a highly explainable feature engineering strategy, that can be applied for a variety of tasks such as sentiment analysis.

See [wiki](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

## Theory
The formula for TF-IDF is as follows:
$$tf(t, d) = \frac{f_{t, d}} {\sum_{t^\prime \in d}{f_{t^\prime,d}}}$$
$$idf(t, D) = \log \frac{|N|} {|d: d \in D \space and \space t \in d |}$$
$$tfidf(t, d, D) = TF(t, d) * IDF(t, D)$$

Where:
* $D$: collection of documents (corpus)
* $d$: a document
* $t$: a term (e.g. word)
* $f_{t, d}$: number of a given term in a given document
* $N$: total number of documents

Essentially, if a term appears often in a document, $tf$ goes up. If a term appears infrequently accross the corpus, $idf$ goes down. Therefore, a high $tfidf$ indicates that the given term is important in that document.

This formula produces a vector for each document, where each element of the vector is the $tfidf$ score for each term.
