from jax import Array
from jax import numpy as jnp


def cartesian_product(*variables: Array) -> Array:
    """
    cartesian_product Return the Cartesian product of input 1-D arrays.

    Parameters
    ----------
    *variables : Array
        One-dimensional arrays whose Cartesian product is desired.

    Returns
    -------
    Array
        2-D array of shape (prod(len(vars)), num_vars) with the Cartesian product.
    """
    num_vars = len(variables)
    meshgrid = jnp.meshgrid(*variables, indexing="ij")
    cart_product = jnp.stack(meshgrid, axis=num_vars).reshape(-1, num_vars)
    return cart_product
