from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import distrax
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


class DistributionalValueCritic(nn.Module):
    """
    A value critic that predicts a Gaussian distribution over returns.
    Attributes:
        ob_dim: Dimensionality of the observation vector.
        n_layers: Number of hidden layers in the MLP.
        layer_size: The width (number of units) of each hidden layer.
    """
    ob_dim: int
    n_layers: int
    layer_size: int

    def setup(self):
        self.value_net = jtu.build_mlp(
            input_size=self.ob_dim,
            output_size=2,  # one for mean and one for log variance
            n_layers=self.n_layers,
            size=self.layer_size,
        )

    def __call__(self, obs: jnp.ndarray) -> distrax.Distribution:
        """
        Forward pass: given an observation, return a Gaussian distribution over values.
        """
        outputs = self.value_net(obs)
        mean, logvar = jnp.split(outputs, 2, axis=-1)
        std = jnp.exp(0.5 * logvar)
        return distrax.Normal(loc=jnp.squeeze(mean, axis=-1), scale=jnp.squeeze(std, axis=-1))

    def create_train_state(self, rng, learning_rate: float) -> train_state.TrainState:
        """
        Initializes network parameters and returns a TrainState.
        """
        tx = optax.adam(learning_rate)
        dummy_input = jnp.ones([1, self.ob_dim])
        params = self.init(rng, dummy_input)
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=tx,
        )

    @partial(jax.jit, static_argnums=0)
    def get_value(self, obs: jnp.ndarray, params: dict) -> jnp.ndarray:
        """
        Returns a deterministic value prediction (the mean of the distribution)
        for the given observation.
        """
        dist = self.apply(params, obs)
        return dist.mean()  # type: ignore

    @partial(jax.jit, static_argnums=0)
    def sample_value(self, obs: jnp.ndarray, params: dict, rng_key) -> jnp.ndarray:
        """
        Returns a stochastic value prediction by sampling from the distribution.
        """
        dist = self.apply(params, obs)
        return dist.sample(seed=rng_key) # type: ignore

    @staticmethod
    @jax.jit
    def update(
        state: train_state.TrainState,
        obs: jnp.ndarray,
        q_values: jnp.ndarray  # target values
    ) -> tuple[train_state.TrainState, jnp.ndarray]:
        """
        Update the critic's parameters using the negative log likelihood loss.
        """
        def loss_fn(params):
            # Get the predicted distribution
            dist = state.apply_fn(params, obs)
            # Compute the negative log likelihood of the target q_values under the predicted distribution
            loss = -jnp.mean(dist.log_prob(q_values))
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
