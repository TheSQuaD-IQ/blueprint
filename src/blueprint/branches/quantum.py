from typing import Tuple
from functools import partial

import jax
from jax import Array
from jax import jit
from jax import numpy as jnp

from ..util.index import get_max_overlap_inds


def get_next_ind(overlap_mat: Array, start_ind: Array) -> Tuple[Array, Array]:
    """
    get_next_ind Find the next branch index given an overlap matrix and a start index.

    Parameters
    ----------
    overlap_mat : Array
        Overlap matrix where entry (i, j) gives overlap metric used for
        selecting indices.
    start_ind : Array
        Index (or indices) used to select the column from which to choose the
        next index.

    Returns
    -------
    Tuple[Array, Array]
        Tuple containing the updated overlap matrix (with the chosen row set to
        zero) and the chosen next index.
    """
    next_ind = jnp.argmax(overlap_mat[:, start_ind])
    overlap_mat = overlap_mat.at[next_ind].set(0.0)
    return overlap_mat, next_ind


def get_next_inds(
    prev_carry: Tuple[Array, Array], _
) -> Tuple[Tuple[Array, Array], Array]:
    """
    get_next_inds Compute the next set of branch indices for a scan carry.

    Parameters
    ----------
    prev_carry : tuple
        Tuple containing the overlap matrix and the previous branch indices.
    _ : None
        Placeholder to fit the ``jax.lax.scan`` function signature.

    Returns
    -------
    Tuple[Tuple[Array, Array], Array]
        Updated carry and the next branch indices.
    """
    overlap_mat, prev_inds = prev_carry
    overlap_mat, next_inds = jax.lax.scan(get_next_ind, overlap_mat, prev_inds)
    next_carry = (overlap_mat, next_inds)
    return next_carry, next_inds


@partial(jit, static_argnames=("num_branches"))
def assign_branch_inds(
    overlap_mat: Array, ground_inds: Array, num_branches: int
) -> Array:
    """
    assign_branch_inds Assign indices to each branch of the resonator using overlaps.

    Parameters
    ----------
    overlap_mat : Array
        The overlap matrix used to decide branch assignment.
    ground_inds : Array
        Indices corresponding to the ground branch states.
    num_branches : int
        Number of resonator branches. Typically the dimensionality of the
        resonator (must be an integer).

    Returns
    -------
    Array
        Array of branch indices with shape ``(num_branches, n_ground)`` before
        raveling.
    """
    overlap_mat = overlap_mat.at[ground_inds].set(0)
    init = (overlap_mat, ground_inds)
    length = num_branches - 1
    _, exc_inds = jax.lax.scan(get_next_inds, init, xs=None, length=length)
    branch_inds = jnp.vstack((ground_inds, exc_inds))
    return branch_inds


def get_branch_inds(states: Array, raise_op: Array, ground_inds: Array) -> Array:
    """
    get_branch_inds Return indices of states grouped into resonator branches.

    Parameters
    ----------
    states : Array
        Eigenstates of the system with shape ``(dim, num_states)`` where each
        column is a state vector.
    raise_op : Array
        Raising operator for the resonator expanded to the system Hilbert
        space (square matrix of shape ``(dim, dim)``).
    ground_inds : Array
        Indices of the product-state vectors corresponding to the resonator's
        ground states.

    Returns
    -------
    Array
        Ravelled indices that select branch-ordered states from the state
        matrix.

    Raises
    ------
    ValueError
        If inputs do not have expected shapes or sizes.
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
    overlap_mat = jnp.abs(
        jnp.einsum("ia, ij, jb -> ab", jnp.conj(states), raise_op, states)
    )

    # number of branches must be an integer
    num_branches = int(dim // num_inds)
    # Assign the remaining branch indices
    inds = assign_branch_inds(overlap_mat, ground_inds, num_branches)

    # Flatten the branch indices to a vector that can be used to extract the branch vectors
    raveled_inds = jnp.ravel(jnp.transpose(inds))

    return raveled_inds


def get_branches(
    hamiltonian: Array,
    raise_op: Array,
    prod_states: Array,
) -> Tuple[Array, Array]:
    """
    get_branch_states Returns the branch energies and eigenstates of the input system Hamiltonian.
    The system is assumed to be composed of a qubit and a resonator, in that order.
    The dressed states are assigned sequentially, starting from the ground branch, corresponding to the resonator being in the ground state.
    The assignment is done by maximizing the overlap of the dressed states with the states obtained from applying the raising operator on the previous branch, starting from the ground branch.

    ----------
    hamiltonian : Array
        The Hamiltonian of the system.
    raise_op : Array
        The raising operator of the resonator, expanded to match the Hilbert space of the system.
    prod_states : int
        The product states of the system and the resonator ground states.
        These are used to sort the dressed states, from which the branches are built.

    Returns
    -------
    Tuple[Array, Array]
        The branch energies and eigenstates of the system.
    """
    energies, states = jnp.linalg.eigh(hamiltonian)

    # Normalize the energies by the ground state energy
    energies = energies - energies[0]

    # Find the ground state indices
    ground_inds = get_max_overlap_inds(prod_states, states)

    # Assign the remaining branch indices
    inds = get_branch_inds(states, raise_op, ground_inds)
    del ground_inds

    # Get the branch energies and vectors
    branch_energies = energies[inds]
    branches = states[:, inds]

    return branch_energies, branches
