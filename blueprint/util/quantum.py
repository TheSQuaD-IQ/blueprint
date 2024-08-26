from jax import numpy as jnp
from jax import Array


def to_denstiy_matrix(state_vec: Array) -> Array:
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
    conj_vec = state_vec.conj()
    density_mat = jnp.einsum("i, j-> ij", state_vec, conj_vec)
    return density_mat


def state_overlap(state: Array, target_state: Array) -> Array:
    """
    state_overlaps Calculates the overlaps of a state with a target state. 'state' and 'target_state' must be either a vector or an array of column vectors, such that the i-th eigenstate corresponds to state[:, i] or target_state[:, i], respectively.

    Parameters
    ----------
    state : Array
        The state(s) that you want to calculate the overlap for.
    target_state : Array
        The target state(s) that you want to calculate the overlap with.

    Returns
    -------
    Array
        The overlap(s) between the state(s) and the target state(s).
    """
    if state.ndim == 0:
        raise ValueError("Input must be a vector or matrix")

    states = jnp.atleast_2d(state)

    if target_state.ndim == 0:
        raise ValueError("Input must be a vector or matrix")

    target_states = jnp.atleast_2d(target_state)

    vec_prod = jnp.einsum("...ij, ...ik -> ...jk", jnp.conj(states), target_states)
    overlap = jnp.abs(vec_prod) ** 2
    return jnp.squeeze(overlap)


def expectation_value(states: Array, operator: Array) -> Array:
    """
    get_expectation_val Calculates the expectation value of an operator given a set of state vectors.

    Parameters
    ----------
    vectors : Array
        The eigenvectors of the Hamiltonian.
    operator : Array
        The operator for which the expectation value is to be calculated.

    Returns
    -------
    Array
        The expectation value of the operator for each state.
    """
    return jnp.einsum("ia, ...ij, ja -> a...", jnp.conj(states), operator, states)
