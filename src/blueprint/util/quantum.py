from jax import numpy as jnp
from jax import Array


def to_denstiy_matrix(state_vec: Array) -> Array:
    """
    to_denstiy_matrix Convert a state vector to a density matrix.

    Parameters
    ----------
    state_vec : Array
        State vector with shape ``(dim,)`` or ``(dim, 1)``.

    Returns
    -------
    Array
        Density matrix with shape ``(dim, dim)``.
    """
    state_vec = jnp.squeeze(state_vec)
    density_mat = jnp.einsum("i, j-> ij", state_vec, jnp.conj(state_vec))
    return density_mat


def state_overlap(states: Array, target_states: Array) -> Array:
    """
    state_overlap Compute squared overlaps between columns of two state matrices.

    Parameters
    ----------
    states : Array
        State matrix with shape ``(dim, n)``; columns are states.
    target_states : Array
        Target state matrix with shape ``(dim, m)``; columns are target states.

    Returns
    -------
    Array
        Matrix of squared overlaps with shape ``(n, m)`` (or squeezed).
    """
    vec_prod = jnp.einsum("ij, ik -> jk", jnp.conj(states), target_states)
    overlap = jnp.abs(vec_prod) ** 2
    return jnp.squeeze(overlap)


def expectation_value(states: Array, operator: Array) -> Array:
    """
    expectation_value Compute expectation values of an operator for each column state.

    Parameters
    ----------
    states : Array
        State matrix with shape ``(dim, n)``.
    operator : Array
        Operator matrix with shape ``(dim, dim)``.

    Returns
    -------
    Array
        Expectation values for each state (shape ``(n,)``).
    """
    return jnp.einsum("ia, ij, ja -> a", jnp.conj(states), operator, states)
