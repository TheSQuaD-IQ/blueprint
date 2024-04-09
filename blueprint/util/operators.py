import math
from typing import Tuple, Any

from jax import Array
from jax import numpy as jnp


def get_pauli_ops(
    normalize: bool = False, dtype: Any | None = None
) -> Tuple[Array, Array, Array, Array]:
    """
    get_pauli_ops Returns the qubit Pauli operators, more specifically in the order of I, X, Y, Z.

    Returns
    -------
    Tuple[Array]
        The Pauli operators I, X, Y, Z (in that order).
    """
    if dtype is not None:
        try:
            dtype = jnp.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"Invalid datatype {dtype}.") from exc
    else:
        dtype = jnp.complex64

    prefactor = 1 / math.sqrt(2) if normalize else 1

    pauli_i = prefactor * jnp.array([[1, 0], [0, 1]], dtype=dtype)
    pauli_x = prefactor * jnp.array([[0, 1], [1, 0]], dtype=dtype)
    pauli_y = prefactor * jnp.array([[0, -1j], [1j, 0]], dtype=dtype)
    pauli_z = prefactor * jnp.array([[1, 0], [0, -1]], dtype=dtype)

    # NOTE: could be a dictionary instead, but this seems more useful.
    return pauli_i, pauli_x, pauli_y, pauli_z
