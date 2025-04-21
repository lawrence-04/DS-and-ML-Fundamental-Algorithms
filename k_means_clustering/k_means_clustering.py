from __future__ import annotations
import numpy as np

class KMeansClusterer:
    def __init__(self, k: int=3, max_iters: int=100, tolerance: float =1e-4, centroids: np.ndarray | None = None) -> None:
        """Class for the K-means clustering algorithm
        
        Args:
            k: number of centroids (clusters)
            max_iters: maximum number of iterations (number of centroid updates)
            tolerance: maximum required centroid movement
            centroids (optional): array of intial centroids 
        """
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = centroids
        
        if centroids is not None:
            assert centroids.shape[0] == k, f"Mismatch between {k=} and {centroids=}"

    def fit(self: KMeansClusterer, X: np.ndarray) -> KMeansClusterer:
        self.n_samples, self.n_features = X.shape

        # initialise centroids from randomly sample data if not provided.
        if self.centroids is None:
            random_indices = np.random.choice(self.n_samples, self.k, replace=False)
            self.centroids = X[random_indices]
        else:
            assert X.shape[1] == self.centroids.shape[1], "Centroids and data have mismatched dimensions."

        for i in range(self.max_iters):
            # assign samples to closest centroids
            self.labels = self.predict(X)

            # save current centroids
            old_centroids = self.centroids.copy()

            # recalculate centroids
            self.centroids = self._calculate_centroids(X)

            # check for convergence
            diff = np.linalg.norm(self.centroids - old_centroids)
            if diff < self.tolerance:
                print(f"Converged at iteration {i}")
                return self
        
        print(f"Unable to converge by iteration {i}")
        return self
    
    def _get_distances_to_centroids(self: KMeansClusterer, X: np.ndarray) -> np.ndarray:
        difference = np.expand_dims(X, axis=1) - self.centroids
        distances = np.linalg.norm(difference, axis=2)
        return distances

    def _assign_clusters(self: KMeansClusterer, distances: np.ndarray) -> np.ndarray:
        labels = np.argmin(distances, axis=1)
        return labels

    def _calculate_centroids(self:KMeansClusterer, X: np.ndarray) -> np.ndarray:
        centroids = np.zeros((self.k, self.n_features))
        for idx in range(self.k):
            cluster_points = X[self.labels == idx]
            if len(cluster_points) > 0:
                centroids[idx] = np.mean(cluster_points, axis=0)
        return centroids

    def predict(self:KMeansClusterer, X: np.ndarray) -> np.ndarray:
        distances = self._get_distances_to_centroids(X)
        labels = self._assign_clusters(distances)
        return labels
