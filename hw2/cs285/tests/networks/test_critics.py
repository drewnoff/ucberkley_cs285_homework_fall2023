from __future__ import annotations
import pytest
import jax
import jax.numpy as jnp
from flax.training import train_state

from cs285.networks.critics_jax import ValueCritic




@pytest.fixture
def critic():
    """Fixture that returns a ValueCritic instance with typical hyperparameters."""
    ob_dim = 10
    n_layers = 2
    layer_size = 64
    return ValueCritic(
        ob_dim=ob_dim,
        n_layers=n_layers,
        layer_size=layer_size,
    )


@pytest.fixture
def rng():
    """Fixture that provides a reproducible JAX PRNGKey."""
    return jax.random.PRNGKey(0)


def test_create_train_state(critic, rng):
    """
    Test that create_train_state returns a valid TrainState
    with non-empty parameters and optimizer.
    """
    state = critic.create_train_state(rng, learning_rate = 1e-3)
    assert isinstance(state, train_state.TrainState)
    flat_params, _ = jax.tree_util.tree_flatten(state.params)
    for param in flat_params:
        assert param is not None


def test_forward(critic, rng):
    """
    Test that the forward pass produces an output of the expected shape.
    The module's __call__ should return a (batch_size,) array given input shape (batch_size, ob_dim).
    """
    state = critic.create_train_state(rng, learning_rate = 1e-3)
    batch_size = 3
    dummy_obs = jnp.ones((batch_size, critic.ob_dim))

    predictions = critic.apply(state.params, dummy_obs)

    assert predictions.shape == (batch_size,)


def test_update(critic, rng):
    """
    Test that a single update step returns a new state and a non-negative loss.
    """
    state = critic.create_train_state(rng, learning_rate = 1e-3)
    batch_size = 5
    dummy_obs = jnp.ones((batch_size, critic.ob_dim))
    dummy_q_values = jnp.ones((batch_size,))

    new_state, loss = ValueCritic.update(state, dummy_obs, dummy_q_values)

    assert isinstance(loss, jnp.ndarray)
    assert loss >= 0


def test_parameters_update(critic, rng):
    """
    Test that after one update step, the network parameters have changed.
    """
    state = critic.create_train_state(rng, learning_rate = 1e-3)
    dummy_obs = jnp.ones((5, critic.ob_dim))
    dummy_q_values = jnp.ones((5,))

    old_params = state.params
    new_state, _ = ValueCritic.update(state, dummy_obs, dummy_q_values)

    def compute_diff(old, new):
        return jnp.sum(jnp.abs(old - new))

    diff_tree = jax.tree_util.tree_map(compute_diff, old_params, new_state.params)
    total_diff = sum(jax.tree_util.tree_leaves(diff_tree))

    assert total_diff > 0
