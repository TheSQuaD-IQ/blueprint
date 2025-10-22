from typing import Tuple
from itertools import accumulate
from operator import mul

import jax
from jax import Array
from jax import numpy as jnp

from .quantum import state_overlap


def assign_max_overlap_ind(overlaps: Array, ind: Array) -> Tuple[Array, Array]:
    """
    assign_max_overlap_ind Assigns the index of the state with the maximum overlap with the target state.

    Parameters
    ----------
    overlaps : Array
        The array of state overlaps between the states and the target states.
    ind : Array
        The index of the state for which the maximum overlap is to be found.

    Returns
    -------
    Tuple[Array, Array]
        The updated array of state overlaps and the index of the state with the maximum overlap.
    """
    max_overlap_ind = jnp.argmax(overlaps[ind])
    filtered_overlaps = overlaps.at[:, max_overlap_ind].set(0)
    return filtered_overlaps, max_overlap_ind


def get_max_overlap_inds(states: Array, target_states: Array) -> Array:
    """
    max_overlap_inds Returns the indices of the states with the maximum overlap with the target states.

    Parameters
    ----------
    states : Array
        The states to compare.
    target_states : Array
        The target states.

    Returns
    -------
    Array
        The indices of the states with the maximum overlap with the target states.
    """
    _, num_states = states.shape

    state_overlaps = state_overlap(states, target_states)
    state_inds = jnp.arange(num_states)

    _, max_overlap_inds = jax.lax.scan(
        assign_max_overlap_ind, state_overlaps, state_inds
    )
    return max_overlap_inds
