from jax import numpy as jnp
from jax import Array

def to_operator(vector: Array) -> Array:
    """
    to_operator Converts a vector to a density matrix operator.

    Parameters
    ----------
    vector : Array
        The state vector.

    Returns
    -------
    Array
        The density matrix operator.
    """
    conj_vector = vector.conj()
    density_mat = jnp.einsum("i, j-> ij", vector, conj_vector)
    return density_mat