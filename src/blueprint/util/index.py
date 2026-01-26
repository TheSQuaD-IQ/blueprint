from typing import Tuple

import jax
from jax import Array
from jax import numpy as jnp

from .quantum import state_overlap


def assign_max_overlap_ind(overlaps: Array, ind: Array) -> Tuple[Array, Array]:
    """
    assign_max_overlap_ind Select the index with maximum overlap and zero that column.

    Parameters
    ----------
    overlaps : Array
        Array of overlaps between states and target states (shape ``(N, M)``).
    ind : Array
        Index or indices selecting which column(s) to inspect.

    Returns
    -------
    Tuple[Array, Array]
        Tuple of (updated_overlaps, max_index) where the chosen column in
        ``updated_overlaps`` has been zeroed.
    """
    max_overlap_ind = jnp.argmax(overlaps[ind])
    filtered_overlaps = overlaps.at[:, max_overlap_ind].set(0)
    return filtered_overlaps, max_overlap_ind


def get_max_overlap_inds(states: Array, target_states: Array) -> Array:
    """
    get_max_overlap_inds Return indices of states that maximize overlap with target states.

    Parameters
    ----------
    states : Array
        Matrix of states with shape ``(dim, num_states)`` (columns are states).
    target_states : Array
        Target state vectors with compatible shape.

    Returns
    -------
    Array
        Indices of the best-matching states for each target state.
    """
    _, num_states = states.shape

    state_overlaps = state_overlap(states, target_states)
    state_inds = jnp.arange(num_states)

    _, max_overlap_inds = jax.lax.scan(
        assign_max_overlap_ind, state_overlaps, state_inds
    )
    return max_overlap_inds
