# Tree-structured Parzen Estimator (TPE)
The Tree-structured Parzen Estimator (TPE) is a Bayesian optimization algorithm commonly used for hyperparameter tuning in machine learning. Unlike grid or random search, TPE intelligently explores the hyperparameter space to find better configurations more efficiently.

## Theory
In hyperparameter optimization, we aim to find a set of hyperparameters that minimizes (or maximizes) an objective function. While simple methods like grid search exhaustively try every combination, this becomes computationally expensive as the number of hyperparameters grows. TPE offers a more efficient alternative by modeling the search space probabilistically.

Instead of directly modeling the objective function as a function of the hyperparameters (p(y | x)), TPE models the inverse: the probability of a hyperparameter configuration given a performance value (p(x | y)). Specifically, it models two separate distributions:

* $l(x) = p(x | y < y*)$, the distribution of good hyperparameters (typically the top 10–20%)

* $g(x) = p(x | y ≥ y*)$, the distribution of bad hyperparameters

These are estimated using non-parametric kernel density estimators (KDE), allowing flexible modeling of complex distributions.

The algorithm proceeds as follows:
1. Initialize with a random set of hyperparameters and evaluate them on the objective function.
1. Split the evaluations into two sets based on a quantile threshold (e.g., top 15% as "good").
1. Fit KDEs to the good and bad hyperparameter sets, denoted as $l(x)$ and $g(x)$.
1. Sample new candidates from l(x) and choose the one that maximizes the ratio $\frac{l(x)}{g(x)}$, approximating expected improvement.
1. Evaluate this candidate and add it to the dataset.
1. Repeat from step 2.

TPE also supports tree-structured or conditional hyperparameter spaces. For example, enabling certain hyperparameters only if a specific model type is selected, making it especially useful for complex search spaces.

For simplicity, we only consider continuous hyperparameters in this implementation, but categorical hyperparameters can be optimised with this technique.
