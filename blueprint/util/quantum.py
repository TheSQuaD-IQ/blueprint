from jax import numpy as jnp
from jax import Array


def to_denstiy_mat(state_vec: Array) -> Array:
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
        _description_
    target_state : Array
        _description_

    Returns
    -------
    Array
        _description_
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
