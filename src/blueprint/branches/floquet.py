from typing import Callable, Tuple

import jax
from jax import Array
from jax import jit
from jax import numpy as jnp
from jaxtyping import Scalar

import dynamiqs as dq
from dynamiqs import Options
from dynamiqs.method import Method, Tsit5

from ..util.index import get_max_overlap_inds

type Pulse = Callable[[float], Scalar | Array]


@jit
def assign_branch_inds(prev_modes: Array, next_modes: Array) -> Tuple[Array, Array]:
    """
    assign_branch_inds Assigns the branch indices for the given modes based
    on their overlap with the previous modes, which are assumed to have been sorted.

    Parameters
    ----------
    prev_modes : Array
        The previous floquet modes of the driven hamiltonian. These are assumed to
        be sorted already.
    next_modes : Array
        The next floquet modes of the driven hamiltonian. These are not sorted.
        The sorting is done based on the overlap with the previous modes.

    Returns
    -------
    Tuple[Array, Array]
        The sorted modes and the sorting indices.
    """
    sort_inds = get_max_overlap_inds(prev_modes, next_modes)
    sorted_modes = next_modes[:, sort_inds]
    return sorted_modes, sort_inds


def get_branch_inds(modes: Array, states: Array) -> Array:
    """
    get_branch_inds Assigns the branch indices for the given modes and states.

    Parameters
    ----------
    modes : Array
        The final floquet modes of the driven hamiltonian for the different
        drive amplitudes considered in the simulation.
    states : Array
        The eigenstates of the driven system. # ? Should this not be the undriven system?

    Returns
    -------
    Array
        The sorting indices of the modes and quasienergies that are adiabatic
        with respect to the increasing drive amplitude.
    """
    _, inds = jax.lax.scan(assign_branch_inds, states, modes)
    return inds


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
    get_branches Calculates the branches of the driven system using the Floquet method.

    Parameters
    ----------
    hamiltonian : Array
        The hamiltonian of the system, excluding the drive.
    drive_pulse : Pulse
        The pulse of the drive. This is a callable function/object
        that takes a single float argument corresponding to time and returns the array of the drive amplitudes at that time.
        The floquet assignment is adabatic with respect to the drive amplitudes.
        This does not multiple drive periods. For batching over periods, see jax.vmap.
    drive_operator : Array
        The operator of the drive.
    drive_period : float
        The period of the drive.
    init_time : float | Scalar, optional
        The initial time for which to simulate the system, by default 0.0
    method : Method | None, optional
        The dynamiqs integration method, by default None which defaults to the Tsit5 method.
    options : Options | None, optional
        The dynamiqs solver options, by default None. If None, the default options are used. See dynamiqs.floquet for the valid options available for the method used.

    Returns
    -------
    Tuple[Array, Array]
        The sorted final floquet modes and quasienergies of the driven system.
        The modes are sorted according to the branch indices, which are assigned based on the overlap with the previous modes as a function of the drive power.
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
