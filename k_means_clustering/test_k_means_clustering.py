import numpy as np
import pytest
from k_means_clustering import KMeansClusterer  # Replace with actual import path

# Fix seed for reproducibility
np.random.seed(42)


def test_basic_clustering():
    # Generate simple 2D data with 3 clusters
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.60, random_state=0)

    kmeans = KMeansClusterer(k=3)
    kmeans.fit(X)

    # Ensure we get labels for each data point
    labels = kmeans.predict(X)
    assert labels.shape == (150,)

    # Ensure we have exactly k unique clusters
    assert len(np.unique(labels)) == 3

    # Ensure centroids are in expected shape
    assert kmeans.centroids.shape == (3, X.shape[1])


def test_convergence():
    X = np.array([[1, 1], [1.1, 1.1], [0.9, 0.9], [10, 10], [10.1, 10.1], [9.9, 9.9]])

    kmeans = KMeansClusterer(k=2)
    kmeans.fit(X)

    assert kmeans.centroids.shape == (2, 2)
    assert (
        np.linalg.norm(kmeans.centroids[0] - kmeans.centroids[1]) > 1.0
    )  # Well separated


def test_custom_centroids():
    X = np.array([[0, 0], [10, 10]])
    centroids = np.array([[0, 0], [10, 10]])

    model = KMeansClusterer(k=2, centroids=centroids)
    model.fit(X)

    # Should converge immediately
    assert np.allclose(model.centroids, centroids)


def test_empty_cluster_reinitialization(monkeypatch):
    # Force an empty cluster
    X = np.array([[1, 1], [1, 1], [1, 1]])
    centroids = np.array([[1, 1], [0, 0]])

    model = KMeansClusterer(k=2, centroids=centroids)

    model.fit(X)
    assert model.centroids.shape == (2, 2)


def test_dimension_mismatch():
    X = np.array([[1, 2], [3, 4]])
    bad_centroids = np.array([[1, 2, 3]])

    with pytest.raises(AssertionError):
        KMeansClusterer(k=1, centroids=bad_centroids).fit(X)
