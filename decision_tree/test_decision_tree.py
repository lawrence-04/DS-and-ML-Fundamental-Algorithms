import numpy as np
from decision_tree import DecisionTreeBinaryClassifier


def test_compute_gini_impurity():
    clf = DecisionTreeBinaryClassifier()
    assert clf.compute_gini_impurity(np.array([0, 0, 0, 0])) == 0.0
    assert clf.compute_gini_impurity(np.array([1, 1, 1, 1])) == 0.0
    assert np.isclose(clf.compute_gini_impurity(np.array([0, 1])), 0.5)


def test_predict_perfectly_separable():
    # XOR-style data (but linearly separable)
    X = np.array([[1], [2], [8], [9]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeBinaryClassifier(max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert np.array_equal(preds, y)


def test_not_enough_samples_for_split():
    X = np.array([[1], [2]])
    y = np.array([0, 1])

    clf = DecisionTreeBinaryClassifier(max_depth=3, min_samples_per_split=2)
    clf.fit(X, y)
    preds = clf.predict(X)

    # With only 2 points and min_samples_per_split=2, tree will not split
    assert all(pred in [0, 1] for pred in preds)
    assert len(set(preds)) == 1  # both predictions should be the same


def test_predict_single_point():
    X = np.array([[1], [2], [3], [10]])
    y = np.array([0, 0, 0, 1])

    clf = DecisionTreeBinaryClassifier(max_depth=2)
    clf.fit(X, y)

    pred = clf.predict(np.array([[10]]))
    assert pred[0] == 1


def test_model_accuracy_on_easy_data():
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    clf = DecisionTreeBinaryClassifier(max_depth=3)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    acc = accuracy_score(y, y_pred)
    assert acc > 0.9
