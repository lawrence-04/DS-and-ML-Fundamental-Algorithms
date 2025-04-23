import numpy as np
from scipy import optimize


class SVMPrimal:
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        optimise_method="L-BFGS-B",
    ):
        """
        Initialize SVM in primal form with soft-margin.

        Args:
        C: Regularization parameter. Trades off margin size and training error.
        max_iter: Maximum number of iterations for the optimizer.
        tol: Tolerance for stopping criterion.
        """
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
        self.optimise_method = optimise_method

    def _objective_function(
        self, params: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        n_features = X.shape[1]
        w = params[:n_features]
        b = params[n_features]

        reg_term = 0.5 * np.dot(w, w)

        margins = y * (np.dot(X, w) + b)
        hinge_losses = np.maximum(0, 1 - margins)
        hinge_term = self.C * np.sum(hinge_losses)

        loss = reg_term + hinge_term
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        initial_params = np.zeros(n_features + 1)

        result = optimize.minimize(
            fun=self._objective_function,
            x0=initial_params,
            args=(X, y),
            method=self.optimise_method,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )

        self.w = result.x[:n_features]
        self.b = result.x[n_features]

        return self

    def predict(self, X):
        decision_values = np.dot(X, self.w) + self.b
        preds = np.sign(decision_values)
        return preds
