from typing import Tuple, Any

from jax import Array
from jax import numpy as jnp

def get_pauli_ops(datatype: Any | None = None) -> Tuple[Array]:
    """
    get_pauli_ops Returns the qubit Pauli operators, more specifically in the order of I, X, Y, Z.

    Returns
    -------
    Tuple[Array]
        The Pauli operators I, X, Y, Z (in that order).
    """
    if datatype is not None:
        try:
            dtype = jnp.dtype(datatype)
        except TypeError as exc:
            raise ValueError(f"Invalid datatype {datatype}.") from exc
    else:
        dtype = jnp.complex64

    identity = jnp.array([[1, 0], [0, 1]], dtype=dtype)
    pauli_x = jnp.array([[0, 1], [1, 0]], dtype=dtype)
    pauli_y = jnp.array([[0, -1j], [1j, 0]], dtype=dtype)
    pauli_z = jnp.array([[1, 0], [0, -1]], dtype=dtype)
    
    #NOTE: could be a dictionary instead, but this seems more useful.
    return identity, pauli_x, pauli_y, pauli_z