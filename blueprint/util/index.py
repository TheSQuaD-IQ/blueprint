from typing import Tuple
from itertools import accumulate
from operator import mul

from jax import Array
from jax import numpy as jnp

from .quantum import state_overlap


def state_index(state: Tuple[int, ...], dims: Tuple[int, ...]) -> int:
    """
    state_index Returns the index of a state in a Hilbert space.

    Parameters
    ----------
    state : Tuple[int]
        The state in the Hilbert space.
    dims : Tuple[int]
        The dimensions of the Hilbert space.

    Returns
    -------
    int
        The index of the state in the Hilbert space.
    """
    reversed_dims = reversed(dims)
    reversed_state = reversed(state)

    exp_dims = (1, *reversed_dims)
    dim_prods = accumulate(exp_dims, mul)

    ind = sum(state * dim for state, dim in zip(reversed_state, dim_prods))

    return ind


def max_overlap_inds(states: Array, target_states: Array) -> Array:
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
    overlaps = state_overlap(states, target_states)
    inds = jnp.argmax(overlaps, axis=-1)

    return inds
