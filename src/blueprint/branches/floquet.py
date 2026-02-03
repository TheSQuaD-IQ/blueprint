from typing import Tuple
from functools import partial

import jax
from jax import Array
from jax import jit
from jax import numpy as jnp
from jaxtyping import Scalar

import dynamiqs as dq
from dynamiqs import Options
from dynamiqs.method import Method, Tsit5

from ..drives import Pulse
from ..util.index import get_max_overlap_inds


@jit
def assign_branch_inds(prev_modes: Array, next_modes: Array) -> Tuple[Array, Array]:
    """
    assign_branch_inds Sort `next_modes` to match ordering of `prev_modes` using overlaps.

    Parameters
    ----------
    prev_modes : Array
        Reference modes (assumed sorted).
    next_modes : Array
        Modes to be sorted according to overlap with `prev_modes`.

    Returns
    -------
    Tuple[Array, Array]
        Tuple of (sorted_modes, sort_indices).
    """
    sort_inds = get_max_overlap_inds(prev_modes, next_modes)
    sorted_modes = next_modes[:, sort_inds]
    return sorted_modes, sort_inds


@jit
def get_branch_inds(modes: Array, states: Array) -> Array:
    """
    get_branch_inds Compute branch sorting indices for Floquet modes across drive amplitudes.

    Parameters
    ----------
    modes : Array
        Floquet modes for each drive amplitude.
    states : Array
        Reference eigenstates used to initialize sorting.

    Returns
    -------
    Array
        Sorting indices for modes and quasienergies.
    """
    _, inds = jax.lax.scan(assign_branch_inds, states, modes)
    return inds


@partial(jit, static_argnames=("method", "options"))
def get_branches(
    hamiltonian: Array,
    drive_pulse: Pulse,
    drive_op: Array,
    drive_period: float,
    init_time: float | Scalar = 0.0,
    method: Method | None = None,
    options: Options | None = None,
) -> Tuple[Array, Array]:
    """
    get_branches Compute Floquet branches (modes and quasienergies) for a driven system.

    Parameters
    ----------
    hamiltonian : Array
        Static part of the system Hamiltonian (shape ``(d,d)``).
    drive_pulse : callable
        Time-dependent drive amplitude function.
    drive_op : Array
        Operator coupled by the drive (shape ``(d,d)``).
    drive_period : float
        Period of the drive.
    init_time : float or Scalar, optional
        Initial time for the Floquet calculation.
    method : Method or None, optional
        Integration method for dynamiqs; defaults to Tsit5.
    options : Options or None, optional
        Solver options for dynamiqs.

    Returns
    -------
    Tuple[Array, Array]
        Tuple of (branch_quasienergies, branches) where branches are sorted modes.
    """
    method = method or Tsit5()
    options = options or Options()

    save_times = jnp.atleast_1d(init_time)

    hamiltonian_term = dq.constant(hamiltonian)
    drive_term = dq.modulated(drive_pulse, drive_op)

    driven_hamiltonian = hamiltonian_term + drive_term

    result = dq.floquet(
        driven_hamiltonian,
        drive_period,
        save_times,
        method=method,
        options=options,
    )

    quasienergies = result.quasienergies
    modes = result.modes.to_jax()
    # Remove the redundant axis corresponding to the time and the one used for the right-hand vectors
    modes = jnp.squeeze(modes)
    # Move the last two axes so that the the modes are the column vectors of the array
    modes = jnp.moveaxis(modes, -1, -2)

    _, states = jnp.linalg.eigh(hamiltonian)
    inds = get_branch_inds(modes, states)

    branch_quasienergies = jnp.take_along_axis(quasienergies, inds, -1)

    exp_inds = jnp.expand_dims(inds, -2)
    branches = jnp.take_along_axis(modes, exp_inds, -1)

    return branch_quasienergies, branches
