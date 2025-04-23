import numpy as np


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.std = None

    def _normalise_features(self, X: np.ndarray, eps=1e-8) -> np.ndarray:
        X_normalised = (X - self.mean) / (self.std + eps)
        return X_normalised

    def fit(self, X: np.ndarray) -> None:
        # normalise
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalised = self._normalise_features(X)

        # covariance matrix
        cov_matrix = np.cov(X_normalised, rowvar=False)

        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvectors by eigenvalues in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idxs]
        eigenvalues = eigenvalues[idxs]

        # select top n_components
        self.components = eigenvectors[:, : self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        # project data onto principal components
        X_normalised = self._normalise_features(X)
        X_transformed = np.dot(X_normalised, self.components)
        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed
