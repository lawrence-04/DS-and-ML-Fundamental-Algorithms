import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformer import Transformer


@pytest.fixture
def sample_model():
    return Transformer(d_model=64, n_heads=8, n_layers=2, d_ff=256, output_size=10)


@pytest.fixture
def sample_data():
    batch_size, seq_len, d_model = 4, 20, 64
    x = torch.randn(batch_size, seq_len, d_model)
    y = torch.randint(0, 10, (batch_size, seq_len))
    return x, y


def test_forward_pass_shape(sample_model, sample_data):
    x, _ = sample_data
    output = sample_model(x)

    batch_size, seq_len = x.shape[:2]
    expected_shape = (batch_size, seq_len, 10)
    assert output.shape == expected_shape


def test_training_loop(sample_model, sample_data):
    x, y = sample_data

    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(sample_model.parameters(), lr=0.001)

    # test fit method runs without error
    sample_model.fit(train_loader, criterion, optimizer, epochs=2)
    assert sample_model.training


def test_inference(sample_model, sample_data):
    x, _ = sample_data

    predictions = sample_model.predict(x)

    assert predictions.shape == (x.shape[0], x.shape[1], 10)
    assert not sample_model.training  # should be in eval mode


def test_gradient_computation(sample_model, sample_data):
    x, y = sample_data

    output = sample_model(x)
    loss = nn.CrossEntropyLoss()(output.view(-1, 10), y.view(-1))
    loss.backward()

    # check gradients exist
    has_gradients = any(
        p.grad is not None for p in sample_model.parameters() if p.requires_grad
    )
    assert has_gradients


@pytest.mark.parametrize("batch_size,seq_len", [(1, 5), (8, 50), (16, 100)])
def test_different_input_sizes(sample_model, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 64)
    output = sample_model(x)
    assert output.shape == (batch_size, seq_len, 10)
