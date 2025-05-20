import math
from itertools import combinations
from typing import Callable

import numpy as np


class Shapley:
    def __init__(self, model_predictor: Callable, X_train: np.ndarray):
        self.model_predictor = model_predictor
        self.X_train = X_train

        self.n_features = X_train.shape[1]
        self.means = np.mean(X_train, axis=0)
        self.n_outputs = model_predictor(X_train[[0]]).shape[1]

        self.mask_lens = []

    def _get_value_for_single_feature(
        self, single_input: np.ndarray, feature_idx: int
    ) -> np.ndarray:
        shapley_value = np.zeros(self.n_outputs)

        for k in range(self.n_features):
            for subset_mask in combinations(range(self.n_features), k):
                if feature_idx in subset_mask:
                    continue

                subset_with_feature = single_input.copy()
                subset_without_feature = single_input.copy()

                # mask out unused features
                subset_with_feature[[subset_mask]] = self.means[[subset_mask]]

                # maske out unused features and active feature
                subset_without_feature[[subset_mask]] = self.means[[subset_mask]]
                subset_without_feature[feature_idx] = self.means[feature_idx]

                value_with_feature = self.model_predictor(
                    subset_with_feature.reshape(1, -1)
                )[0]
                value_without_feature = self.model_predictor(
                    subset_without_feature.reshape(1, -1)
                )[0]

                n_subset = self.n_features - len(subset_mask) - 1
                normalisation_factor = (
                    math.factorial(n_subset)
                    * math.factorial(self.n_features - n_subset - 1)
                    / math.factorial(self.n_features)
                )

                shapley_value += normalisation_factor * (
                    value_with_feature - value_without_feature
                )

        return shapley_value

    def get_values(self, single_input: np.ndarray) -> np.ndarray:
        shapley_values = np.zeros((self.n_features, self.n_outputs))

        for i in range(self.n_features):
            shapley_values[i] = self._get_value_for_single_feature(
                single_input=single_input, feature_idx=i
            )

        return shapley_values
