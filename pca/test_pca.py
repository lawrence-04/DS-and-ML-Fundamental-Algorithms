import numpy as np
from numpy.testing import assert_almost_equal

from pca import PCA


def test_pca_shapes():
    # Generate random data
    np.random.seed(42)
    X = np.random.rand(100, 5)

    # Instantiate and run PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    # Assert output shape is correct
    assert X_transformed.shape == (100, 2)


def test_pca_variance_direction():
    # Create data with clear dominant direction
    np.random.seed(42)
    X = np.dot(np.random.rand(100, 2), np.array([[3, 1], [1, 0.5]]).T)

    # Fit PCA
    pca = PCA(n_components=1)
    X_transformed = pca.fit_transform(X)

    # Transformed data should have zero mean along the component axis
    assert np.allclose(np.mean(X_transformed, axis=0), 0, atol=1e-7)


def test_pca_reconstruction_approx():
    # Test if projection and inverse approximate original (for full components)
    np.random.seed(42)
    X = np.random.rand(50, 3)

    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X)

    # Reconstruct normalized X
    X_normalized = (X - pca.mean) / (pca.std + 1e-8)
    X_approx = X_transformed @ pca.components.T

    # Should approximately match normalized X
    assert_almost_equal(X_approx, X_normalized, decimal=5)
