import pytest
import numpy as np
from unittest.mock import MagicMock
from q_learning import QLearning  # Adjust if in different file


def step_fn():
    # Each episode has 3 steps and ends on the third
    steps = [
        (np.array([0.1, 0.0, 0.02, 0.0]), 1.0, False, False, {}),
        (np.array([0.2, 0.0, 0.04, 0.0]), 1.0, False, False, {}),
        (np.array([0.3, 0.0, 0.06, 0.0]), 1.0, True,  False, {}),
    ]
    while True:
        for step in steps:
            yield step

@pytest.fixture
def mock_env():
    env = MagicMock()
    env.action_space.n = 2
    env.action_space.sample.return_value = 1
    env.reset.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), {})
    env.step.side_effect = step_fn()

    return env

def test_initialization(mock_env):
    agent = QLearning(mock_env)
    assert agent.q_table.shape == (agent.n_discrete_states, agent.n_actions)
    assert agent.epsilon == 1.0
    assert isinstance(agent.bins, list)
    assert len(agent.bins) == 4

def test_discretize_state(mock_env):
    agent = QLearning(mock_env)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    discrete_state = agent.discretize_state(state)
    assert isinstance(discrete_state, int)
    assert 0 <= discrete_state < agent.n_discrete_states

def test_select_action_exploration(mock_env):
    agent = QLearning(mock_env)
    agent.epsilon = 1.0  # Force exploration
    state = agent.discretize_state(np.array([0.0, 0.0, 0.0, 0.0]))
    action = agent.select_action(state)
    assert action in [0, 1]

def test_select_action_exploitation(mock_env):
    agent = QLearning(mock_env)
    agent.epsilon = 0.0  # Force exploitation
    state = agent.discretize_state(np.array([0.0, 0.0, 0.0, 0.0]))
    agent.q_table[state, 0] = 0.5
    agent.q_table[state, 1] = 1.0
    action = agent.select_action(state)
    assert action == 1

def test_update_q_table(mock_env):
    agent = QLearning(mock_env)
    state = agent.discretize_state(np.array([0.0, 0.0, 0.0, 0.0]))
    next_state = agent.discretize_state(np.array([0.1, 0.0, 0.02, 0.0]))
    action = 1
    reward = 1.0
    done = False
    old_value = agent.q_table[state, action]
    agent.update_q_table(state, action, reward, next_state, done)
    new_value = agent.q_table[state, action]
    assert new_value != old_value

def test_decay_epsilon(mock_env):
    agent = QLearning(mock_env, epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.1)
    agent.decay_epsilon()
    assert agent.epsilon == 0.5
    agent.decay_epsilon()
    assert agent.epsilon == 0.25
    agent.decay_epsilon()
    agent.decay_epsilon()
    agent.decay_epsilon()
    agent.decay_epsilon()
    agent.decay_epsilon()
    assert agent.epsilon >= 0.1  # Should not drop below epsilon_min

def test_train_returns_results(mock_env):
    agent = QLearning(mock_env)
    rewards, lengths = agent.train(n_episodes=5)
    assert len(rewards) == 5
    assert len(lengths) == 5
