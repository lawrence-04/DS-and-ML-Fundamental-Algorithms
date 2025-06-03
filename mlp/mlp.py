import numpy as np


class MLP:
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        lr: float = 0.01,
    ) -> None:
        """
        Initialize a simple 2-layer MLP.

        Args:
            num_inputs: Number of input features.
            num_hidden: Number of hidden units.
            num_outputs: Number of output units.
            lr: Learning rate for gradient descent.
        """
        self.W1: np.ndarray = np.random.randn(num_hidden, num_inputs) * 0.01
        self.b1: np.ndarray = np.zeros((num_hidden, 1))
        self.W2: np.ndarray = np.random.randn(num_outputs, num_hidden) * 0.01
        self.b2: np.ndarray = np.zeros((num_outputs, 1))
        self.lr: float = lr

        self.losses: list[float] = []

    @staticmethod
    def relu(X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    @staticmethod
    def grad_relu(X: np.ndarray) -> np.ndarray:
        return (X > 0).astype(float)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = self.W1 @ X + self.b1  # n_hidden x n_samples
        self.a1 = self.relu(self.z1)  # n_hidden x n_samples
        self.z2 = self.W2 @ self.a1 + self.b2  # n_outputs x n_samples
        return self.z2

    def backward(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
        m = y.shape[1]  # number of samples

        delta2 = y_hat - y  # shape (n_outputs, n_samples)
        grad_W2 = (delta2 @ self.a1.T) / m  # n_outputs x n_hidden
        grad_b2 = np.mean(delta2, axis=1, keepdims=True)

        delta1 = (self.W2.T @ delta2) * self.grad_relu(self.z1)
        grad_W1 = (delta1 @ X.T) / m
        grad_b1 = np.mean(delta1, axis=1, keepdims=True)

        # update weights and biases
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        """
        Train the MLP using gradient descent.

        Args:
            X: input data, shape (num_inputs, num_samples)
            y: target data, shape (num_outputs, num_samples)
            epochs: number of training epochs
        """
        # X should be n_inputs x n_samples
        # y should be n_outputs x n_samples
        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = np.mean((y_hat - y) ** 2) / 2
            self.backward(X, y, y_hat)

            self.losses.append(loss)
