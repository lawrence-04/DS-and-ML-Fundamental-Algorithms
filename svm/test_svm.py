import numpy as np
import pytest

from svm import SVMPrimal  # Update this to match the actual file/module name


@pytest.fixture
def linearly_separable_data():
    np.random.seed(42)
    X_pos = np.random.randn(10, 2) + 2
    X_neg = np.random.randn(10, 2) - 2
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(10), -np.ones(10)))
    return X, y


def test_initialization():
    model = SVMPrimal(C=0.5, max_iter=200, tol=1e-3)
    assert model.C == 0.5
    assert model.max_iter == 200
    assert model.tol == 1e-3
    assert model.w is None
    assert model.b is None


def test_fit_and_predict_linearly_separable(linearly_separable_data):
    X, y = linearly_separable_data
    model = SVMPrimal(C=1.0)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)

    assert model.w is not None
    assert model.b is not None
    assert accuracy >= 0.9  # Should classify most correctly


def test_prediction_shape(linearly_separable_data):
    X, y = linearly_separable_data
    model = SVMPrimal()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_zero_data():
    X = np.zeros((4, 2))
    y = np.array([1, -1, 1, -1])
    model = SVMPrimal()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (4,)


def test_invalid_input_shapes(linearly_separable_data):
    X, y = linearly_separable_data
    model = SVMPrimal()
    model.fit(X, y)

    X_invalid = np.random.randn(5, 3)  # Wrong feature dimension
    with pytest.raises(ValueError):
        model.predict(X_invalid)
