from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np

from cs285.infrastructure import jax_util as jtu


class ValueCritic(nn.Module):
    """
    A combined value critic module. This Flax module builds an MLP that maps
    observations to scalar values and also provides helper functions for
    creating a training state and updating parameters.

    Attributes:
        ob_dim: Dimensionality of the observation vector.
        n_layers: Number of hidden layers in the MLP.
        layer_size: The width (number of units) of each hidden layer.
        learning_rate: Learning rate for the Adam optimizer.
    """
    ob_dim: int
    n_layers: int
    layer_size: int

    def setup(self):
        self.value_net = jtu.build_mlp(
            input_size=self.ob_dim,
            output_size=1,
            n_layers=self.n_layers,
            size=self.layer_size,
        )

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: given an observation, return its value prediction.
        The raw network returns shape (batch, 1) so we squeeze the last dimension.
        """
        value = self.value_net(obs)
        return jnp.squeeze(value, axis=-1)

    def create_train_state(self, rng, learning_rate: float) -> train_state.TrainState:
        """
        Initializes network parameters using the given RNG and returns a
        TrainState that bundles the parameters and the optimizer.

        Args:
            rng: A JAX PRNG key. (This should be provided from outside.)

        Returns:
            A train_state.TrainState instance.
        """
        tx = optax.adam(learning_rate)
        dummy_input = jnp.ones([1, self.ob_dim])
        params = self.init(rng, dummy_input)
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=tx,
        )

    @staticmethod
    @jax.jit
    def update(
        state: train_state.TrainState,
        obs: jnp.ndarray,
        q_values: jnp.ndarray
    ) -> tuple[train_state.TrainState, jnp.ndarray]:
        """
        Perform one gradient update step on the critic's parameters using MSE loss.

        Args:
            state: The current training state.
            obs: Batch of observations.
            targets: Batch of target values corresponding to each observation.

        Returns:
            A tuple of (updated_train_state, loss).
        """
        def loss_fn(params):
            preds = state.apply_fn(params, obs)
            loss = jnp.mean((preds - q_values) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
