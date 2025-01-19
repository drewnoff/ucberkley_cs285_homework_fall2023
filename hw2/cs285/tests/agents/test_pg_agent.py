from __future__ import annotations

import pytest
import numpy as np

from cs285.agents.pg_agent_jax import PGAgent, calculate_discounted_return, calculate_discounted_reward_to_go
from cs285.infrastructure.jax_util import to_numpy

OB_DIM = 4
AC_DIM = 2


@pytest.fixture
def basic_agent():
    """Create a basic discrete action space agent for testing."""
    return PGAgent(
        ob_dim=OB_DIM,
        ac_dim=AC_DIM,
        discrete=True,
        n_layers=2,
        layer_size=64,
        learning_rate=1e-3,
        use_reward_to_go=False,
        gamma=0.99,
        normalize_advantages=False,
    )


@pytest.fixture
def continuous_agent():
    """Create a continuous action space agent for testing."""
    return PGAgent(
        ob_dim=OB_DIM,
        ac_dim=AC_DIM,
        discrete=False,
        n_layers=2,
        layer_size=64,
        learning_rate=1e-3,
        use_reward_to_go=True,
        gamma=0.99,
        normalize_advantages=True,
    )


def test_discounted_return_utils():
    rewards = np.array([1.0, 2.0, 3.0])
    gamma = 0.99

    returns = calculate_discounted_return(rewards, gamma)

    expected = 1.0 + 2.0 * gamma + 3.0 * gamma**2
    assert np.allclose(returns, np.array([expected, expected, expected]))


def test_discounted_reward_to_go_utils():
    rewards = np.array([1.0, 2.0, 3.0])
    gamma = 0.99

    rtg = calculate_discounted_reward_to_go(rewards, gamma)

    expected = np.array([1.0 + 2.0 * gamma + 3.0 * gamma**2, 2.0 + 3.0 * gamma, 3.0])
    assert np.allclose(rtg, expected)


def test_calculate_q_vals_no_reward_to_go(basic_agent):
    rewards = [np.array([1.0, 2.0, 3.0])]
    q_vals = basic_agent._calculate_q_vals(rewards)

    expected = calculate_discounted_return(rewards[0], basic_agent.gamma)
    assert np.allclose(q_vals[0], expected)


def test_get_action_discrete(basic_agent):
    """Test action sampling for discrete action space."""
    obs = np.random.random(OB_DIM)
    action = to_numpy(basic_agent.get_action(obs))
    assert action.shape == ()  # Discrete actions should be scalar
    assert action.dtype == np.int32


def test_get_action_continuous(continuous_agent):
    """Test action sampling for continuous action space."""
    obs = np.random.random(OB_DIM)
    action = to_numpy(continuous_agent.get_action(obs))
    assert isinstance(action, np.ndarray)
    assert action.shape == (AC_DIM,)  # Continuous actions should match ac_dim
    assert action.dtype == np.float32


def test_update_basic(basic_agent):
    """Test the update method with basic configuration."""
    T = 3  # num timesteps
    obs = [np.random.random((T, OB_DIM))]  # 3 timesteps, 4 features
    actions = [np.random.randint(0, 2, size=(T,))]
    rewards = [np.arange(1, T + 1).astype(np.float32)]

    info = basic_agent.update(obs, actions, rewards)
    assert 'Actor Loss' in info
    assert info['Actor Loss'].shape == ()
    assert info['Actor Loss'] >= 0.0


def test_multiple_trajectories(basic_agent):
    """Test handling of multiple trajectories."""
    obs = [np.random.random((2, OB_DIM)), np.random.random((3, OB_DIM))]
    actions = [np.random.randint(0, 2, size=(2,)), np.random.randint(0, 2, size=(3,))]
    rewards = [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])]

    info = basic_agent.update(obs, actions, rewards)
    assert 'Actor Loss' in info
    assert info['Actor Loss'].shape == ()
    assert info['Actor Loss'] >= 0.0
