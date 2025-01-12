from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.errors import ScopeParamShapeError
from flax.training import train_state

import distrax

from cs285.networks.policies_jax import MLPPolicy, MLPPolicyPG

SEED = 0


def params_all_close(params_l, params_r):
    return jax.tree_util.tree_all(
            jax.tree.map(
                lambda x, y: jnp.allclose(x, y),
                params_l,
                params_r
            )
        )


class TestMLPPolicy:
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(SEED)

    @pytest.fixture
    def discrete_policy(self):
        return MLPPolicy(ac_dim=3, ob_dim=4, discrete=True, n_layers=2, layer_size=64)

    @pytest.fixture
    def continuous_policy(self):
        return MLPPolicy(ac_dim=2, ob_dim=4, discrete=False, n_layers=2, layer_size=64)

    def test_discrete_policy_init(self, discrete_policy, rng_key):
        rng, _ = jax.random.split(rng_key)
        obs = jnp.ones((1, 4))
        params = discrete_policy.init(rng, obs)

        assert 'params' in params
        assert 'logits_net' in params['params']

    def test_continuous_policy_init(self, continuous_policy, rng_key):
        rng = jax.random.split(rng_key)[0]
        obs = jnp.ones((1, 4))
        params = continuous_policy.init(rng, obs)

        assert 'params' in params
        assert 'mean_net' in params['params']
        assert 'logstd' in params['params']

    def test_discrete_policy_forward(self, discrete_policy, rng_key):
        rng = jax.random.split(rng_key)[0]
        obs = jnp.ones((1, 4))
        params = discrete_policy.init(rng, obs)

        dist = discrete_policy.apply(params, obs)

        assert isinstance(dist, distrax.Categorical)
        assert dist.logits.shape == (1, 3)  # batch_size x ac_dim

    def test_continuous_policy_forward(self, continuous_policy, rng_key):
        rng = jax.random.split(rng_key)[0]
        obs = jnp.ones((1, 4))
        params = continuous_policy.init(rng, obs)

        dist = continuous_policy.apply(params, obs)

        assert isinstance(dist, distrax.MultivariateNormalDiag)
        assert dist.loc.shape == (1, 2)  # batch_size x ac_dim
        assert dist.scale_diag.shape == (1, 2)

    def test_discrete_policy_action(self, discrete_policy, rng_key):
        rng = jax.random.split(rng_key)[0]
        obs = jnp.ones((1, 4))
        params = discrete_policy.init(rng, obs)

        action = discrete_policy.get_action(obs, params, rng)

        assert action.shape == (1,)
        assert action.dtype == jnp.int32
        assert 0 <= action[0] < 3  # action should be within valid range

    def test_continuous_policy_action(self, continuous_policy, rng_key):
        rng = jax.random.split(rng_key)[0]
        obs = jnp.ones((1, 4))
        params = continuous_policy.init(rng, obs)

        action = continuous_policy.get_action(obs, params, rng)

        assert action.shape == (1, 2)
        assert action.dtype == jnp.float32

    def test_train_state_creation(self, discrete_policy, rng_key):
        rng = jax.random.split(rng_key)[0]
        learning_rate = 1e-3

        state = discrete_policy.create_train_state(rng, learning_rate)

        assert hasattr(state, 'params')
        assert hasattr(state, 'apply_fn')
        assert hasattr(state, 'tx')

    def test_gradient_computation(self, continuous_policy, rng_key):
        rng = jax.random.split(rng_key)[0]
        obs = jnp.ones((1, 4))
        params = continuous_policy.init(rng, obs)

        def loss_fn(params):
            dist = continuous_policy.apply(params, obs)
            return -dist.log_prob(jnp.zeros((1, 2))).mean()

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)

        assert 'mean_net' in grads['params']
        assert 'logstd' in grads['params']

    def test_batch_processing(self, continuous_policy, rng_key):
        rng = jax.random.split(rng_key)[0]
        batch_size = 32
        obs = jnp.ones((batch_size, 4))
        params = continuous_policy.init(rng, obs)
        dist = continuous_policy.apply(params, obs)

        assert dist.loc.shape == (batch_size, 2)
        assert dist.scale_diag.shape == (batch_size, 2)

    def test_different_observations(self, continuous_policy, rng_key):
        rng = jax.random.split(rng_key)[0]
        obs1 = jnp.ones((1, 4))
        obs2 = jnp.zeros((1, 4))
        params = continuous_policy.init(rng, obs1)

        dist1 = continuous_policy.apply(params, obs1)
        dist2 = continuous_policy.apply(params, obs2)

        assert not jnp.array_equal(dist1.loc, dist2.loc)

    @pytest.mark.parametrize(
        'invalid_obs',
        [
            jnp.ones((1, 3)),
            jnp.ones((1, 5)),  # wrong input dimension
        ],
    )
    def test_invalid_inputs(self, continuous_policy, rng_key, invalid_obs):
        rng = jax.random.split(rng_key)[0]
        params = continuous_policy.init(rng, jnp.ones((1, 4)))

        with pytest.raises(ScopeParamShapeError):
            continuous_policy.apply(params, invalid_obs)


class TestMLPPolicyPG:
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(SEED)

    @pytest.fixture
    def discrete_policy(self):
        return MLPPolicyPG(ac_dim=3, ob_dim=4, discrete=True, n_layers=2, layer_size=64)

    @pytest.fixture
    def continuous_policy(self):
        return MLPPolicyPG(ac_dim=2, ob_dim=4, discrete=False, n_layers=2, layer_size=64)

    def test_discrete_policy_update(self, discrete_policy, rng_key):
        learning_rate = 1e-3
        state = discrete_policy.create_train_state(rng_key, learning_rate)

        NxT = 32
        obs = np.random.normal(size=(NxT, discrete_policy.ob_dim))
        actions = np.random.randint(0, discrete_policy.ac_dim, size=(NxT,))
        advantages = np.random.normal(size=(NxT,))

        new_state, metrics = discrete_policy.update(state, obs, actions, advantages)

        assert isinstance(new_state, train_state.TrainState)
        assert 'Actor Loss' in metrics
        assert isinstance(metrics['Actor Loss'], (float, np.floating, jnp.ndarray))
        assert not params_all_close(state.params, new_state.params)

    def test_continuous_policy_update(self, continuous_policy, rng_key):
        learning_rate = 1e-3
        state = continuous_policy.create_train_state(rng_key, learning_rate)

        NxT = 32
        obs = np.random.normal(size=(NxT, continuous_policy.ob_dim))
        actions = np.random.normal(size=(NxT, continuous_policy.ac_dim))
        advantages = np.random.normal(size=(NxT,))

        new_state, metrics = continuous_policy.update(state, obs, actions, advantages)

        assert isinstance(new_state, train_state.TrainState)
        assert 'Actor Loss' in metrics
        assert isinstance(metrics['Actor Loss'], (float, np.floating, jnp.ndarray))
        assert not params_all_close(state.params, new_state.params)

    def test_update_with_zero_advantages(self, discrete_policy, rng_key):
        learning_rate = 1e-3
        state = discrete_policy.create_train_state(rng_key, learning_rate)

        NxT = 32
        obs = np.random.normal(size=(NxT, discrete_policy.ob_dim))
        actions = np.random.randint(0, discrete_policy.ac_dim, size=(NxT,))
        advantages = np.zeros(NxT)

        new_state, _ = discrete_policy.update(state, obs, actions, advantages)

        assert params_all_close(state.params, new_state.params)

    def test_update_gradient_scale(self, continuous_policy, rng_key):

        learning_rate = 1e-3
        state = continuous_policy.create_train_state(rng_key, learning_rate)

        NxT = 32
        obs = np.random.normal(size=(NxT, continuous_policy.ob_dim))
        actions = np.random.normal(size=(NxT, continuous_policy.ac_dim))

        small_advantages = np.random.normal(size=(NxT,)) * 0.1
        large_advantages = np.random.normal(size=(NxT,)) * 10.0

        new_state_small, _ = continuous_policy.update(
            state, obs, actions, small_advantages)
        new_state_large, _ = continuous_policy.update(
            state, obs, actions, large_advantages)

        # The larger advantages should lead to larger parameter changes
        small_param_change = jax.tree.map(
            lambda x, y: jnp.abs(x - y).mean(),
            new_state_small.params, state.params
        )
        large_param_change = jax.tree.map(
            lambda x, y: jnp.abs(x - y).mean(),
            new_state_large.params, state.params
        )
        # Compare the magnitude of parameter changes
        assert jax.tree_util.tree_all(
            jax.tree.map(lambda x, y: x < y, small_param_change, large_param_change)
        )

    def test_update_batch_size_invariance(self, discrete_policy, rng_key):
        learning_rate = 1e-3
        state = discrete_policy.create_train_state(rng_key, learning_rate)

        batch_sizes = [16, 32, 64]
        losses = []

        for batch_size in batch_sizes:
            obs = np.random.normal(size=(batch_size, discrete_policy.ob_dim))
            actions = np.random.randint(0, discrete_policy.ac_dim, size=(batch_size,))
            advantages = np.random.normal(size=(batch_size,))

            _, metrics = discrete_policy.update(state, obs, actions, advantages)
            losses.append(metrics['Actor Loss'])

        loss_diffs = [abs(losses[i] - losses[i+1]) for i in range(len(losses)-1)]
        assert all(diff < 1.0 for diff in loss_diffs)  # threshold can be adjusted
