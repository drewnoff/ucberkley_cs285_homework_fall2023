from __future__ import annotations

import jax
from typing import Callable
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

Activation = str | Callable

_str_to_activation = {
    'relu': nn.relu,
    'tanh': nn.tanh,
    'leaky_relu': lambda x: nn.leaky_relu(x, negative_slope=0.01),
    'sigmoid': nn.sigmoid,
    'selu': nn.selu,
    'softplus': nn.softplus,
    'identity': lambda x: x,
}


class MLP(nn.Module):
    input_size: int
    output_size: int
    n_layers: int
    size: int
    activation: Activation = 'tanh'
    output_activation: Activation = 'identity'

    @nn.compact
    def __call__(self, x):
        activation_fn = _str_to_activation[self.activation] if isinstance(self.activation, str) else self.activation

        if isinstance(self.output_activation, str):
            output_activation_fn = _str_to_activation[self.output_activation]
        else:
            output_activation_fn = self.output_activation

        for _ in range(self.n_layers):
            x = nn.Dense(features=self.size)(x)
            x = activation_fn(x)

        x = nn.Dense(features=self.output_size)(x)
        return output_activation_fn(x)

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
    Creates an MLP model instance.
    """
    return MLP(
        input_size=input_size,
        output_size=output_size,
        n_layers=n_layers,
        size=size,
        activation=activation,
        output_activation=output_activation
    )

def from_numpy(x: np.ndarray,):
    return jnp.array(x, dtype=jnp.float32)


def to_numpy(x):
    return jax.device_get(x)


"""
# Example usage:
model = build_mlp(
    input_size=10,
    output_size=5,
    n_layers=2,
    size=64
)

# Initialize parameters with a key
rng = jax.random.PRNGKey(0)
rng, inp_rng, init_rng = jax.random.split(rng, 3)

x = jax.random.normal(inp_rng, (1, 10))
params = model.init(init_rng, x)
output = model.apply(params, x)
"""
