import pytest
import jax
import jax.numpy as jnp
import numpy as np


from cs285.infrastructure.jax_util import build_mlp, from_numpy, to_numpy, _str_to_activation

@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)

@pytest.fixture
def sample_mlp():
    return build_mlp(
        input_size=10,
        output_size=5,
        n_layers=2,
        size=64
    )


def test_mlp_initialization(rng_key, sample_mlp):
    _, inp_rng, init_rng = jax.random.split(rng_key, 3)
    x = jax.random.normal(inp_rng, (1, 10))

    params = sample_mlp.init(init_rng, x)

    assert 'params' in params

    # For 2 hidden layers + output layer, we should have 3 Dense layers
    dense_layers = [k for k in params['params'] if 'Dense' in k]
    assert len(dense_layers) == 3


def test_mlp_forward_pass(rng_key, sample_mlp):
    _, inp_rng, init_rng = jax.random.split(rng_key, 3)
    x = jax.random.normal(inp_rng, (1, 10))
    params = sample_mlp.init(init_rng, x)

    # Test forward pass
    output = sample_mlp.apply(params, x)

    # Check output shape
    assert output.shape == (1, 5)


def test_different_activations(rng_key):
    for activation in _str_to_activation:
        mlp = build_mlp(
            input_size=10,
            output_size=5,
            n_layers=2,
            size=64,
            activation=activation
        )

        _, inp_rng, init_rng = jax.random.split(rng_key, 3)
        x = jax.random.normal(inp_rng, (1, 10))
        params = mlp.init(init_rng, x)
        output = mlp.apply(params, x)

        assert output.shape == (1, 5) # type: ignore


def test_from_numpy():
    # Test with different numpy array shapes
    test_arrays = [
        np.random.rand(10),
        np.random.rand(10, 10),
        np.random.rand(10, 10, 10)
    ]

    for arr in test_arrays:
        jax_arr = from_numpy(arr)
        assert isinstance(jax_arr, jnp.ndarray)
        assert jax_arr.shape == arr.shape
        assert jnp.allclose(jax_arr, arr)


def test_to_numpy():
    test_arrays = [
        jax.random.normal(jax.random.PRNGKey(0), (10,)),
        jax.random.normal(jax.random.PRNGKey(1), (10, 10)),
        jax.random.normal(jax.random.PRNGKey(2), (10, 10, 10))
    ]

    for arr in test_arrays:
        np_arr = to_numpy(arr)
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == arr.shape
        assert np.allclose(np_arr, arr)


def test_mlp_batch_processing(rng_key, sample_mlp):
    _, inp_rng, init_rng = jax.random.split(rng_key, 3)
    # Test with different batch sizes
    batch_sizes = [1, 10, 100]

    for batch_size in batch_sizes:
        x = jax.random.normal(inp_rng, (batch_size, 10))
        params = sample_mlp.init(init_rng, x)
        output = sample_mlp.apply(params, x)

        assert output.shape == (batch_size, 5)


def test_mlp_parameter_shapes(rng_key):
    input_size = 10
    hidden_size = 64
    output_size = 5
    n_layers = 2

    mlp = build_mlp(
        input_size=input_size,
        output_size=output_size,
        n_layers=n_layers,
        size=hidden_size
    )

    _, inp_rng, init_rng = jax.random.split(rng_key, 3)
    x = jax.random.normal(inp_rng, (1, input_size))
    params = mlp.init(init_rng, x)

    dense_layers = [v for k, v in params['params'].items() if 'Dense' in k]

    # First layer
    assert dense_layers[0]['kernel'].shape == (input_size, hidden_size) # type: ignore
    assert dense_layers[0]['bias'].shape == (hidden_size,) # type: ignore

    # Hidden layers
    assert dense_layers[1]['kernel'].shape == (hidden_size, hidden_size) # type: ignore
    assert dense_layers[1]['bias'].shape == (hidden_size,) # type: ignore

    # Output layer
    assert dense_layers[2]['kernel'].shape == (hidden_size, output_size) # type: ignore
    assert dense_layers[2]['bias'].shape == (output_size,) # type: ignore


def test_deterministic_output(rng_key, sample_mlp):
    rng, inp_rng, init_rng = jax.random.split(rng_key, 3)
    x = jax.random.normal(inp_rng, (1, 10))
    params = sample_mlp.init(init_rng, x)

    # Multiple forward passes should give same result
    output1 = sample_mlp.apply(params, x)
    output2 = sample_mlp.apply(params, x)

    assert jnp.allclose(output1, output2)
