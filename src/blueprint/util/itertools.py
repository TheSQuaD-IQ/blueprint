from jax import Array
from jax import numpy as jnp


def cartesian_product(*variables: Array) -> Array:
    """
    cartesian_product Returns the Cartesian product of the input variables.
    Each input variable is expected to be a vector.

    Returns
    -------
    Array
        The Cartesian product of the input variables.
    """
    num_vars = len(variables)
    meshgrid = jnp.meshgrid(*variables, indexing="ij")
    cart_product = jnp.stack(meshgrid, axis=num_vars).reshape(-1, num_vars)
    return cart_product
