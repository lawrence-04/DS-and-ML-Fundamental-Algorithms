import numpy as np

from mlp import MLP  # replace 'your_module' with the actual filename without .py


def test_initialization_shapes():
    mlp = MLP(3, 4, 2)
    assert mlp.W1.shape == (4, 3)
    assert mlp.b1.shape == (4, 1)
    assert mlp.W2.shape == (2, 4)
    assert mlp.b2.shape == (2, 1)


def test_relu_and_grad_relu():
    x = np.array([-1, 0, 2, -0.5, 3])
    relu = MLP.relu(x)
    grad = MLP.grad_relu(x)
    np.testing.assert_array_equal(relu, [0, 0, 2, 0, 3])
    np.testing.assert_array_equal(grad, [0, 0, 1, 0, 1])


def test_forward_output_shape():
    mlp = MLP(3, 5, 2)
    x = np.random.randn(3, 10)
    y_hat = mlp.forward(x)
    assert y_hat.shape == (2, 10)


def test_backward_updates_weights():
    mlp = MLP(2, 3, 1, lr=0.1)
    x = np.random.randn(2, 5)
    y = np.random.randn(1, 5)

    old_W1 = mlp.W1.copy()
    old_b1 = mlp.b1.copy()
    old_W2 = mlp.W2.copy()
    old_b2 = mlp.b2.copy()

    y_hat = mlp.forward(x)
    mlp.backward(x, y, y_hat)

    # weights and biases should change after backward
    assert not np.allclose(mlp.W1, old_W1)
    assert not np.allclose(mlp.b1, old_b1)
    assert not np.allclose(mlp.W2, old_W2)
    assert not np.allclose(mlp.b2, old_b2)


def test_fit_reduces_loss():
    np.random.seed(0)
    mlp = MLP(2, 5, 1, lr=0.1)
    x = np.random.randn(2, 20)
    y = (x[0:1, :] * 2 + x[1:2, :] * -3 + 1).reshape(1, 20)  # linear target

    mlp.fit(x, y, epochs=50)
    assert len(mlp.losses) == 50
    assert mlp.losses[-1] < mlp.losses[0]
