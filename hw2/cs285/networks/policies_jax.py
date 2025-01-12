from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import numpy as np
import optax
from functools import partial
import distrax
from cs285.infrastructure import jax_util as jtu


class MLPPolicy(nn.Module):
    """Base MLP policy that maps observations to distributions over actions.

    Attributes:
        ac_dim: Dimension of the action space
        ob_dim: Dimension of the observation space
        discrete: If True, outputs categorical distribution for discrete actions.
                If False, outputs multivariate normal for continuous actions.
        n_layers: Number of hidden layers in the MLP
        layer_size: Size of each hidden layer
    """
    ac_dim: int
    ob_dim: int
    discrete: bool
    n_layers: int
    layer_size: int

    def setup(self):
        """Initialize the policy networks based on action space type."""
        if self.discrete:
            self.logits_net = jtu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.layer_size,
            )
        else:
            self.mean_net = jtu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.layer_size,
            )
            self.logstd = self.param('logstd',
                                   lambda _: jnp.zeros(self.ac_dim))

    def __call__(self, obs) -> distrax.Distribution:
        """Forward pass of the policy network.

        Args:
            obs: Observation input to the policy

        Returns:
            distrax.Distribution: Either Categorical (discrete) or MultivariateNormalDiag (continuous)
                                representing the policy's action distribution
        """
        if self.discrete:
            logits = self.logits_net(obs)
            return distrax.Categorical(logits=logits)
        else:  # noqa: RET505
            mean = self.mean_net(obs)
            std = jnp.exp(self.logstd)
            return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

    @partial(jax.jit, static_argnums=0)
    def get_action(self, obs: np.ndarray, params: dict, rng_key) -> np.ndarray:
        """Samples a single action from the policy for the given observation.

        Note: This function is not differentiable due to the sampling operation.

        Args:
            obs: Single observation
            params: Policy parameters
            rng_key: JAX random key for sampling

        Returns:
            np.ndarray: Sampled action
        """
        dist = self.apply(params, obs)
        action = dist.sample(seed=rng_key) # type: ignore
        return action

    def create_train_state(self, rng_key, learning_rate: float) -> train_state.TrainState:
        """Creates initial `TrainState`."""
        params = self.init(rng_key, jnp.ones([1, self.ob_dim]))
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=tx,
        )

    def update(
            self,
            state: train_state.TrainState,
            obs: np.ndarray,
            actions: np.ndarray,
            *args,
            **kwargs
    ) -> tuple[train_state.TrainState, dict]:
        """Base update method to be implemented by subclasses."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy class for the policy gradient algorithm."""

    @jax.jit
    def update(
        self,
        state: train_state.TrainState,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> tuple[train_state.TrainState, dict]:
        """Implements the policy gradient actor update."""
        obs = jtu.from_numpy(obs) # type: ignore
        actions = jtu.from_numpy(actions) # type: ignore
        advantages = jtu.from_numpy(advantages) # type: ignore

        def loss_fn(params):
            # TODO: implement the policy gradient loss
            loss = None
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        metrics = {
            'Actor Loss': loss,
        }
        return state, metrics
