"""
Floquet branch utilities.

Utilities to compute and sort Floquet modes (branches) and quasienergies for driven systems.
"""

import jax
from jax import Array
from jax import numpy as jnp
from jax import jit
from jaxtyping import Scalar

import dynamiqs as dq
from dynamiqs import Options
from dynamiqs.method import Method, Tsit5

from ..drives import Pulse
from ..util.index import get_max_overlap_inds

type Float = float | Scalar


@jit
def assign_branch_inds(prev_modes: Array, next_modes: Array) -> tuple[Array, Array]:
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
    tuple[Array, Array]
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


def get_branches(
    hamiltonian: Array,
    drive_pulse: Pulse,
    drive_op: Array,
    drive_period: Float,
    time: Float | Array = 0.0,
    method: Method | None = None,
    options: Options | None = None,
) -> tuple[Array, Array]:
    """
    get_branches Compute Floquet branches (modes and quasienergies) for a driven system.

    Parameters
    ----------
    hamiltonian : Array
        Static part of the system Hamiltonian (shape ``(d,d)``).
    drive_pulse : Pulse
        Time-dependent drive amplitude function. Expected signature
        `amplitude = drive_pulse(t)` where `t` is a scalar time.
    drive_op : Array
        Operator coupled by the drive (shape ``(d,d)``).
    drive_period : Float
        Period of the drive.
    time : Float | Array, optional
        Initial time for the Floquet calculation.
    method : Method | None, optional
        Integration method for dynamiqs; defaults to `Tsit5()`.
    options : Options | None, optional
        Solver options for dynamiqs.

    Returns
    -------
    tuple[Array, Array]
        Tuple of (branch_quasienergies, branches) where branches are sorted modes.
    """
    method = method or Tsit5()
    options = options or Options()

    time = jnp.asarray(time)
    times = jnp.atleast_1d(time)

    hamiltonian_term = dq.constant(hamiltonian)
    drive_term = dq.modulated(drive_pulse, drive_op)

    driven_hamiltonian = hamiltonian_term + drive_term

    result = dq.floquet(
        driven_hamiltonian,
        drive_period,  # type: ignore
        times,
        method=method,
        options=options,
    )

    quasienergies = result.quasienergies
    modes = result.modes.to_jax()
    # Remove singleton axes that may come from dynamiqs representation

    # Move the last two axes so that the the modes are the column vectors of the array
    modes = jnp.squeeze(modes, -1)
    modes = jnp.matrix_transpose(modes)

    _, states = jnp.linalg.eigh(hamiltonian)

    init_modes = jnp.take(modes, 0, -3)
    inds = get_branch_inds(init_modes, states)

    branch_quasienergies = jnp.take_along_axis(quasienergies, inds, -1)

    exp_inds = jnp.expand_dims(inds, (-2, -3))
    branches = jnp.take_along_axis(modes, exp_inds, -1)

    # Remove the time axis if only a single time was provided
    branches = jnp.squeeze(branches, -3)
    return branch_quasienergies, branches
