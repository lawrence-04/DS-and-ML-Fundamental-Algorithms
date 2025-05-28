from typing import Callable

import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm


class TPEstimator:
    def __init__(
        self,
        hyperparameter_ranges: list[list[float]],
        objective_function: Callable,
        num_init_samples: int = 25,
        num_candidate_samples: int = 100,
        num_iter: int = 50,
        good_threshold: float = 0.2,
    ):
        if int(num_init_samples * good_threshold) < len(hyperparameter_ranges):
            raise ValueError(
                "Number of samples in the 'good' distribution must be greater than the number of hyperparameters to comply with scipy.stats.gaussian_kde."
            )

        self.hyperparameter_ranges = hyperparameter_ranges
        self.objective_function = objective_function
        self.num_init_samples = num_init_samples
        self.num_candidate_samples = num_candidate_samples
        self.num_iter = num_iter
        self.good_threshold = good_threshold

        self.hyperparameters = self._initalise_hyperparameters()
        self.values = self._evaluate_inital_hyperparameters()

    def _initalise_hyperparameters(self) -> np.ndarray:
        hyperparameters = []
        for low, high in self.hyperparameter_ranges:
            hyperparameters.append(
                np.random.uniform(low=low, high=high, size=self.num_init_samples)
            )

        hyperparameters = np.array(hyperparameters).T
        return hyperparameters

    def _evaluate_hyperparameter_set(self, hyperparameter_set: np.ndarray) -> float:
        value = self.objective_function(hyperparameters=hyperparameter_set)
        return value

    def _evaluate_inital_hyperparameters(self) -> np.ndarray:
        values = []
        for hyperparameter_set in tqdm(self.hyperparameters):
            value = self._evaluate_hyperparameter_set(hyperparameter_set)
            values.append(value)

        values = np.array(values)
        return values

    def _build_distributions(self):
        sorted_value_idxs = np.argsort(self.values)

        threshold_idx = int(sorted_value_idxs.shape[0] * self.good_threshold)
        good_idxs = sorted_value_idxs[:threshold_idx]
        bad_idxs = sorted_value_idxs[threshold_idx:]

        # gaussian_kde works with dimension num_features x num_samples, so we transpose
        good_dist = gaussian_kde(self.hyperparameters[good_idxs].T)
        bad_dist = gaussian_kde(self.hyperparameters[bad_idxs].T)

        return good_dist, bad_dist

    def _sample_new_candidate(self, good_dist, bad_dist) -> np.ndarray:
        good_candidates = good_dist.resample(self.num_candidate_samples).T

        good_candidates = good_candidates.clip(
            min=[i[0] for i in self.hyperparameter_ranges],
            max=[i[1] for i in self.hyperparameter_ranges],
        )

        good_scores = good_dist(good_candidates.T)
        bad_scores = bad_dist(good_candidates.T)

        ratios = good_scores / bad_scores.clip(min=1e-6)

        best_ratio_idx = np.argmin(ratios)
        best_candidate = good_candidates[best_ratio_idx]
        return best_candidate

    def optimise(self) -> np.ndarray:
        for _ in tqdm(range(self.num_iter)):
            good_dist, bad_dist = self._build_distributions()
            new_candidate = self._sample_new_candidate(
                good_dist=good_dist, bad_dist=bad_dist
            )

            new_candidate_value = self._evaluate_hyperparameter_set(new_candidate)

            self.hyperparameters = np.vstack([self.hyperparameters, new_candidate])
            self.values = np.append(self.values, new_candidate_value)

        best_value_idx = np.argmin(self.values)
        best_hyperparameters = self.hyperparameters[best_value_idx]
        return best_hyperparameters
