from typing import Tuple
from functools import partial
from itertools import accumulate
from operator import mul

import jax
from jax import Array
from jax import numpy as jnp

from .quantum import state_overlap
from ..util.linalg import dag


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
    state_dim, num_states = states.shape
    target_state_dim, num_target_states = target_states.shape

    if state_dim != target_state_dim:
        raise ValueError("Input states must have the same dimension")

    if num_states > num_target_states:
        raise ValueError(
            "The number of target states must be greater than the number of states"
        )

    state_overlaps = state_overlap(states, target_states)
    state_inds = jnp.arange(num_states)

    _, max_overlap_inds = jax.lax.scan(
        assign_max_overlap_ind, state_overlaps, state_inds
    )
    return max_overlap_inds


def get_next_ind(overlap_mat: Array, start_ind: Array) -> Tuple[Array, Array]:
    """
    get_next_ind Finds the next branch index given the current branch index and an overlap matrix.
    The overlap matrix holds the information between

    Parameters
    ----------
    overlap_mat : Array

    start_ind : Array
        _description_

    Returns
    -------
    Tuple[Array, Array]
        The updated overlap matrix and the next branch index.
    """
    next_ind = jnp.argmax(overlap_mat[:, start_ind])
    overlap_mat = overlap_mat.at[next_ind].set(0.0)
    return overlap_mat, next_ind


def get_next_inds(
    prev_carry: Tuple[Array, Array], _
) -> Tuple[Tuple[Array, Array], Array]:
    """
    get_next_inds Finds the set of branch indices for the next branch given the current branch indices.
    Each of the set of branch indicies correspond to the resonator being fixed in a certain state.

    Parameters
    ----------
    prev_carry : Tuple[Array, Array]
        The overlap matrix and the previous branch indices.
    _ : None
        Useless argument introduced to fit the signature of `jax.lax.scan`.

    Returns
    -------
    Tuple[Tuple[Array, Array], Array]
        The updated carry (overlap matrix and next branch indices) and the next branch indices (which are passed again for saving).
    """
    overlap_mat, prev_inds = prev_carry
    overlap_mat, next_inds = jax.lax.scan(get_next_ind, overlap_mat, prev_inds)
    next_carry = (overlap_mat, next_inds)
    return next_carry, next_inds


@partial(jax.jit, static_argnames=("num_branches"))
def assign_branch_inds(
    overlap_mat: Array, ground_inds: Array, num_branches: int
) -> Array:
    """
    assign_branch_inds Assigns an index to each branch of the resonator.

    Parameters
    ----------
    overlap_mat : Array
        The overlap matrix.
    ground_inds : Array
        The indices of the ground state.
    num_branches : int
        The number of resonator branches. Typically, this is the dimensionality of the resonator.

    Returns
    -------
    Array
        The branch indices.
    """
    overlap_mat = overlap_mat.at[ground_inds].set(0)
    init = (overlap_mat, ground_inds)
    length = num_branches - 1
    _, exc_inds = jax.lax.scan(get_next_inds, init, xs=None, length=length)
    branch_inds = jnp.vstack((ground_inds, exc_inds))
    return branch_inds


def get_branch_inds(states: Array, raise_op: Array, ground_inds: Array) -> Array:
    """
    get_branch_inds Returns the indices of the states with maximum overlap with the states obtained by applying the raising operator to the input states.

    Parameters
    ----------
    states : Array
        The eigenstates of the system.
    raise_op : Array
        The raising operator of the resonator.
    ground_inds : Array
        The indices of the states corresponding to the resonator being in the ground state.

    Returns
    -------
    Array
        The indices of each of the branches of the resonator.

    Raises
    ------
    ValueError
        If the states are not a matrix of column vectors.
    ValueError
        If the number of states is greater than the dimension of the Hilbert space.
    ValueError
        If the raising operator is not a square matrix.
    ValueError
        If the raising operator does not have the same dimension as the Hilbert space of the states.
    ValueError
        If the number of ground states is greater than the dimension of the Hilbert space.
    """
    num_dims = len(states.shape)
    if num_dims != 2:
        raise ValueError("The states must be a matrix of column vectors.")

    num_states, dim = states.shape

    if num_states > dim:
        raise ValueError(
            "The number of states cannot be greater than the dimension of the Hilbert space."
        )

    num_op_dims = len(raise_op.shape)
    if num_op_dims != 2:
        raise ValueError("The raising operator must be a matrix.")

    op_dim, other_dim = raise_op.shape
    if op_dim != other_dim:
        raise ValueError("The raising operator must be a square matrix.")

    if op_dim != dim:
        raise ValueError(
            "The raising operator must have the same dimension as the Hilbert space of the states."
        )

    num_inds = len(ground_inds)

    if num_inds >= dim:
        raise ValueError(
            "The number of ground states must be less than the dimension of the Hilbert space."
        )

    # Compute the branch criteria matrix
    overlap_mat = jnp.abs(dag(states) @ raise_op @ states)

    num_branches = dim / num_inds
    # Assign the remaining branch indices
    inds = assign_branch_inds(overlap_mat, ground_inds, num_branches)

    # Flatten the branch indices to a vector thaat can be used to extract the branch vectors
    raveled_inds = jnp.ravel(jnp.transpose(inds))

    return raveled_inds


def get_branch_states(
    hamiltonian: Array, raise_op: Array, res_dim: int
) -> Tuple[Array, Array]:
    """
    get_branch_states Returns the branch energies and eigenstates of the input system Hamiltonian.
    The system is assumed to be composed of a qubit and a resonator, in that order.
    The dressed states are assigned sequentially, starting from the ground state branch,
    corresponding to the resonator being in the ground state.
    The assignment is done by maximizing the overlap of the dressed states with the states
    resulting from the action of the raising operator on the previously identified states,
    starting from the ground branch.

    ----------
    hamiltonian : Array
        The Hamiltonian of the system.
    raise_op : Array
        The raising operator of the resonator, expanded to match the Hilbert space of the system.
    res_dim : int
        The dimension of the resonator Hilbert space.

    Returns
    -------
    Tuple[Array, Array]
        The branch energies and eigenstates of the system.
    """
    energies, states = jnp.linalg.eigh(hamiltonian)  # Normal scipy way

    # Normalize the energies by the ground state energy
    energies = energies - energies[0]

    # Compute the branch criteria matrix
    overlap_mat = jnp.abs(dag(states) @ raise_op @ states)

    # Find the ground state indices
    ground_inds = jnp.argmax(jnp.abs(states[::res_dim]), axis=1)

    # Assign the remaining branch indices
    inds = assign_branch_inds(overlap_mat, ground_inds, res_dim)

    # Flatten the branch indices to a vector thaat can be used to extract the branch vectors
    raveled_inds = jnp.ravel(jnp.transpose(inds))

    # Get the branch energies and vectors
    energies = energies[raveled_inds]
    states = states[:, raveled_inds]

    return energies, states
