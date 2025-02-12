from __future__ import annotations
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from flax.training import train_state
import distrax

from cs285.networks.critics_jax import DistributionalValueCritic, ValueCritic

CRITIC_CONFIG = {
    "ob_dim": 10,
    "n_layers": 2,
    "layer_size": 64,
}

# Dedicated fixtures for each critic.
@pytest.fixture
def value_critic():
    """Returns a ValueCritic instance."""
    return ValueCritic(**CRITIC_CONFIG) # type: ignore

@pytest.fixture
def dist_critic():
    """Returns a DistributionalValueCritic instance."""
    return DistributionalValueCritic(**CRITIC_CONFIG) # type: ignore

@pytest.fixture(params=[ValueCritic, DistributionalValueCritic])
def critic(request):
    CriticClass = request.param
    return CriticClass(**CRITIC_CONFIG)

@pytest.fixture
def rng():
    """Provides a reproducible JAX PRNGKey."""
    return jax.random.PRNGKey(0)


def test_create_train_state(critic, rng):
    """
    This test runs for both ValueCritic and DistributionalValueCritic.
    """
    state = critic.create_train_state(rng, learning_rate=1e-3)
    assert isinstance(state, train_state.TrainState)
    flat_params, _ = jax.tree_util.tree_flatten(state.params)
    for param in flat_params:
        assert param is not None



def test_forward(value_critic, rng):
    """
    Test that the forward pass produces an output of the expected shape.
    The module's __call__ should return a (batch_size,) array given input shape (batch_size, ob_dim).
    """
    state = value_critic.create_train_state(rng, learning_rate = 1e-3)
    batch_size = 3
    dummy_obs = jnp.ones((batch_size, value_critic.ob_dim))

    predictions = value_critic.apply(state.params, dummy_obs)

    assert predictions.shape == (batch_size,)


def test_update(critic, rng):
    """
    Test that a single update step returns a new state and a non-negative loss.
    """
    state = critic.create_train_state(rng, learning_rate = 1e-3)
    batch_size = 5
    dummy_obs = jnp.ones((batch_size, critic.ob_dim))
    dummy_q_values = jnp.ones((batch_size,))

    new_state, loss = critic.update(state, dummy_obs, dummy_q_values)

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
    new_state, _ = critic.update(state, dummy_obs, dummy_q_values)

    def compute_diff(old, new):
        return jnp.sum(jnp.abs(old - new))

    diff_tree = jax.tree_util.tree_map(compute_diff, old_params, new_state.params)
    total_diff = sum(jax.tree_util.tree_leaves(diff_tree))

    assert total_diff > 0


def test_sample_value(dist_critic, rng):
    """
    Test that sample_value returns a sample from the distribution with the correct shape.
    """
    rng, key = jax.random.split(rng)
    state = dist_critic.create_train_state(key, learning_rate=1e-3)
    batch_size = 4
    dummy_obs = np.ones((batch_size, dist_critic.ob_dim))
    rng, key = jax.random.split(rng)
    sample = dist_critic.sample_value(dummy_obs, state.params, key)
    assert sample.shape == (batch_size,)


def test_sample_value_stochastic(dist_critic, rng):
    """
    Test that sample_value returns different samples when using two different random keys.
    """
    rng, key_init = jax.random.split(rng)
    state = dist_critic.create_train_state(key_init, learning_rate=1e-3)
    batch_size = 4
    dummy_obs = np.ones((batch_size, dist_critic.ob_dim))

    rng, key1 = jax.random.split(rng)
    rng, key2 = jax.random.split(rng)

    sample1 = dist_critic.sample_value(dummy_obs, state.params, key1)
    sample2 = dist_critic.sample_value(dummy_obs, state.params, key2)

    assert not jnp.array_equal(sample1, sample2), "Samples should be different for different random keys."


def test_get_value(dist_critic, rng):
    """
    Test that get_value returns a deterministic value estimate equal to the distribution's mean.
    """
    state = dist_critic.create_train_state(rng, learning_rate=1e-3)
    batch_size = 4
    dummy_obs = np.ones((batch_size, dist_critic.ob_dim))
    value = dist_critic.get_value(dummy_obs, state.params)
    dist = dist_critic.apply(state.params, jnp.asarray(dummy_obs))
    expected_value = dist.mean()
    np.testing.assert_allclose(value, expected_value, atol=1e-5)
