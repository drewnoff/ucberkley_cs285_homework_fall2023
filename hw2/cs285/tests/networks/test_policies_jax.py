import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from flax.errors import ScopeParamShapeError
import distrax

from cs285.networks.policies_jax import MLPPolicy

INIT_RNG = jax.random.PRNGKey(0)


class TestMLPPolicy:
    @pytest.fixture
    def discrete_policy(self):
        return MLPPolicy(ac_dim=3, ob_dim=4, discrete=True, n_layers=2, layer_size=64)

    @pytest.fixture
    def continuous_policy(self):
        return MLPPolicy(ac_dim=2, ob_dim=4, discrete=False, n_layers=2, layer_size=64)

    def test_discrete_policy_init(self, discrete_policy):
        rng, _ = jax.random.split(INIT_RNG)
        obs = jnp.ones((1, 4))
        params = discrete_policy.init(rng, obs)

        assert 'params' in params
        assert 'logits_net' in params['params']

    def test_continuous_policy_init(self, continuous_policy):
        rng = jax.random.split(INIT_RNG)[0]
        obs = jnp.ones((1, 4))
        params = continuous_policy.init(rng, obs)

        assert 'params' in params
        assert 'mean_net' in params['params']
        assert 'logstd' in params['params']

    def test_discrete_policy_forward(self, discrete_policy):
        rng = jax.random.split(INIT_RNG)[0]
        obs = jnp.ones((1, 4))
        params = discrete_policy.init(rng, obs)

        dist = discrete_policy.apply(params, obs)

        assert isinstance(dist, distrax.Categorical)
        assert dist.logits.shape == (1, 3)  # batch_size x ac_dim

    def test_continuous_policy_forward(self, continuous_policy):
        rng = jax.random.split(INIT_RNG)[0]
        obs = jnp.ones((1, 4))
        params = continuous_policy.init(rng, obs)

        dist = continuous_policy.apply(params, obs)

        assert isinstance(dist, distrax.MultivariateNormalDiag)
        assert dist.loc.shape == (1, 2)  # batch_size x ac_dim
        assert dist.scale_diag.shape == (1, 2)

    def test_discrete_policy_action(self, discrete_policy):
        rng = jax.random.split(INIT_RNG)[0]
        obs = jnp.ones((1, 4))
        params = discrete_policy.init(rng, obs)

        action = discrete_policy.get_action(obs, params, rng)

        assert action.shape == (1,)
        assert action.dtype == jnp.int32
        assert 0 <= action[0] < 3  # action should be within valid range

    def test_continuous_policy_action(self, continuous_policy):
        rng = jax.random.split(INIT_RNG)[0]
        obs = jnp.ones((1, 4))
        params = continuous_policy.init(rng, obs)

        action = continuous_policy.get_action(obs, params, rng)

        assert action.shape == (1, 2)
        assert action.dtype == jnp.float32

    def test_train_state_creation(self, discrete_policy):
        rng = jax.random.split(INIT_RNG)[0]
        learning_rate = 1e-3

        state = discrete_policy.create_train_state(rng, learning_rate)

        assert hasattr(state, 'params')
        assert hasattr(state, 'apply_fn')
        assert hasattr(state, 'tx')

    def test_gradient_computation(self, continuous_policy):
        rng = jax.random.split(INIT_RNG)[0]
        obs = jnp.ones((1, 4))
        params = continuous_policy.init(rng, obs)

        def loss_fn(params):
            dist = continuous_policy.apply(params, obs)
            return -dist.log_prob(jnp.zeros((1, 2))).mean()

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)

        assert 'mean_net' in grads['params']
        assert 'logstd' in grads['params']

    def test_batch_processing(self, continuous_policy):
        rng = jax.random.split(INIT_RNG)[0]
        batch_size = 32
        obs = jnp.ones((batch_size, 4))
        params = continuous_policy.init(rng, obs)
        dist = continuous_policy.apply(params, obs)

        assert dist.loc.shape == (batch_size, 2)
        assert dist.scale_diag.shape == (batch_size, 2)

    def test_different_observations(self, continuous_policy):
        rng = jax.random.split(INIT_RNG)[0]
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
    def test_invalid_inputs(self, continuous_policy, invalid_obs):
        rng = jax.random.split(INIT_RNG)[0]
        params = continuous_policy.init(rng, jnp.ones((1, 4)))

        with pytest.raises(ScopeParamShapeError):
            continuous_policy.apply(params, invalid_obs)
