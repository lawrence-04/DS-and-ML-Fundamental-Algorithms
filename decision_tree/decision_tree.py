from __future__ import annotations

import numpy as np


class DecisionTreeBinaryClassifier:
    def __init__(self, max_depth: int = 2, min_samples_per_split: int = 1) -> None:
        """
        Binary decision tree classifier using Gini impurity.

        Args:
            max_depth: maximum depth of the tree
            min_samples_per_split: minimum samples required to consider a split
        """
        self.max_depth = max_depth
        self.min_samples_per_split = min_samples_per_split

        self.left_node: DecisionTreeBinaryClassifier | None = None
        self.right_node: DecisionTreeBinaryClassifier | None = None
        self.feature_index: int | None = None
        self.feature_value: float | None = None
        self.label: int | None = None

    @staticmethod
    def compute_gini_impurity(y: np.ndarray) -> float:
        n_total = y.shape[0]
        n_ones = np.sum(y)
        n_zeros = n_total - n_ones

        gini_impurity = 1 - ((n_ones / n_total) ** 2 + (n_zeros / n_total) ** 2)
        return gini_impurity

    def _compute_split_impurity(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        gini_impurity_left = self.compute_gini_impurity(y_left)
        gini_impurity_right = self.compute_gini_impurity(y_right)

        n_total = y_left.shape[0] + y_right.shape[0]
        split_score = (y_left.shape[0] / n_total) * gini_impurity_left + (
            y_right.shape[0] / n_total
        ) * gini_impurity_right
        return split_score

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int, float, float]:
        lowest_impurity = np.inf
        best_split_value = None
        best_split_feature_index = None

        # loop over each feature
        for feature_index in range(X.shape[1]):
            # find the unique values so we don't repeat checks
            # this will only speed up performance of categorical features
            thresholds = np.unique(X[:, feature_index])

            # loop over and check each split
            for threshold in thresholds:
                mask_left = X[:, feature_index] <= threshold
                mask_right = ~mask_left
                y_left, y_right = y[mask_left], y[mask_right]

                if (y_left.shape[0] < self.min_samples_per_split) or (
                    y_right.shape[0] < self.min_samples_per_split
                ):
                    continue

                split_impurity = self._compute_split_impurity(
                    y_left=y_left, y_right=y_right
                )

                # check if it's the best impurity
                if split_impurity < lowest_impurity:
                    lowest_impurity = split_impurity
                    best_split_value = threshold
                    best_split_feature_index = feature_index

        assert best_split_value is not None, (
            f"Not enough data for splitting with {self.min_samples_per_split=}"
        )

        return best_split_feature_index, best_split_value, lowest_impurity

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> None:
        average_label = np.mean(y)
        if depth == self.max_depth:
            self.label = int(average_label > 0.5)
            return

        if np.isclose(average_label, 0):
            self.label = 0
            return

        if np.isclose(average_label, 1):
            self.label = 1
            return

        if X.shape[0] < (2 * self.min_samples_per_split):
            self.label = int(average_label > 0.5)
            return

        best_split_feature_index, best_split_value, _ = self._find_best_split(X=X, y=y)

        self.feature_index = best_split_feature_index
        self.feature_value = best_split_value

        left_indices = X[:, best_split_feature_index] <= best_split_value
        X_left = X[left_indices, :]
        X_right = X[~left_indices, :]

        y_left = y[left_indices]
        y_right = y[~left_indices]

        self.left_node = DecisionTreeBinaryClassifier(
            max_depth=self.max_depth, min_samples_per_split=self.min_samples_per_split
        )
        self.right_node = DecisionTreeBinaryClassifier(
            max_depth=self.max_depth, min_samples_per_split=self.min_samples_per_split
        )

        self.left_node._build_tree(X=X_left, y=y_left, depth=depth + 1)
        self.right_node._build_tree(X=X_right, y=y_right, depth=depth + 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the decision tree classifier to the data.

        Args:
            X: feature matrix
            y: target labels
        """
        self._build_tree(X=X, y=y)

    def _predict_single(self, x: np.ndarray) -> int:
        if self.label is not None:
            return self.label

        if x[self.feature_index] <= self.feature_value:
            return self.left_node._predict_single(x)

        return self.right_node._predict_single(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for multiple samples.

        Args:
            X: feature matrix

        Returns:
            Array of predicted labels
        """
        preds = np.array([self._predict_single(x) for x in X])
        return preds
