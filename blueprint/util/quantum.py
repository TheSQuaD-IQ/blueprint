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
    state_dim = len(state.shape)
    target_dim = len(target_state.shape)

    out_inds = []
    if state_dim == 1:
        state_sublist = [0]
    else:
        state_sublist = [Ellipsis, 0, 1]
        out_inds.append(1)

    if target_dim == 1:
        target_sublist = [0]
    else:
        target_sublist = [Ellipsis, 0, 2]
        out_inds.append(2)

    out_sublist = [Ellipsis, *out_inds]

    overlap = jnp.einsum(
        state,
        state_sublist,
        target_state,
        target_sublist,
        out_sublist,
    )

    return jnp.abs(overlap)


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
