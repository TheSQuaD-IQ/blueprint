import math
from functools import partial

import jax
from jax import numpy as jnp
from jaxtyping import Array, Scalar

import dynamiqs as dq
from dynamiqs import Options
from dynamiqs.method import Method, Tsit5

from .floquet import get_branch_inds
from ..drives import Pulse

type Float = float | Scalar


def get_propagated_branches(
    hamiltonian: Array,
    drive_pulse: Pulse,
    drive_op: Array,
    drive_period: Float,
    save_times: Array,
    method: Method,
    options: Options,
) -> tuple[Array, Array]:
    hamiltonian_term = dq.constant(hamiltonian)
    drive_term = dq.modulated(drive_pulse, drive_op)

    driven_hamiltonian = hamiltonian_term + drive_term

    result = dq.floquet(
        driven_hamiltonian,
        drive_period,  # type: ignore
        save_times,
        method=method,
        options=options,
    )

    quasienergies = result.quasienergies
    modes = result.modes.to_jax()
    # Remove the redundant axis used for the right-hand vectors
    # Then move the last two axes so that the the modes are the column vectors of the array
    modes = jnp.matrix_transpose(jnp.squeeze(modes, -1))

    # get the eigenstates of the Hamiltonian
    _, states = jnp.linalg.eigh(hamiltonian)

    # Take the mode at the initial time (t=0) for the sorting.
    # The time axis is the second to last axis (-3)
    init_modes = jnp.take(modes, 0, -3)
    inds = get_branch_inds(init_modes, states)
    quasienergies = jnp.take_along_axis(quasienergies, inds, -1)

    exp_inds = jnp.expand_dims(inds, (-2, -3))
    branches = jnp.take_along_axis(modes, exp_inds, -1)
    return quasienergies, branches


def get_floquet_detunings(
    quasienergies: Array,
    drive_photons: Array,
    drive_frequency: Float,
) -> Array:
    energy_diffs = quasienergies[:, :, None] - quasienergies[:, None]
    floquet_detunings = energy_diffs[..., None] + drive_photons * drive_frequency
    return floquet_detunings


def get_floquet_matrix_elements(
    modes: Array,
    times: Array,
    drive_op: Array,
    drive_frequency: Float,
    drive_photons: Array,
) -> Array:
    num_times = len(times)
    phases = jnp.exp(-1.0j * drive_frequency * drive_photons * times[:, None])
    integrand_sum = jnp.einsum(
        "atik, ij, atjl, tn -> akln",
        jnp.conj(modes),
        drive_op,
        modes,
        phases,
        optimize=True,
    )
    matrix_elements = integrand_sum / (num_times - 1)
    return matrix_elements


def get_transition_rate(
    frequencies: Array, spectral_density: Array, thermal_populations: Array
) -> Array:
    pos_filter = jnp.heaviside(frequencies, 1.0)
    absorption_rates = thermal_populations * spectral_density * pos_filter
    emission_rates = (1 + thermal_populations) * spectral_density * (1 - pos_filter)
    transition_rates = absorption_rates + emission_rates
    return transition_rates


@partial(jax.jit, static_argnames=["num_times", "num_photons", "method", "options"])
def get_floquet_rates(
    hamiltonian: Array,
    drive_pulse: Pulse,
    drive_op: Array,
    drive_period: Float,
    spectral_density: Array,
    thermal_populations: Array,
    num_times: int,
    num_photons: int,
    method: Method | None = None,
    options: Options | None = None,
) -> tuple[Array, Array, Array]:
    method = method or Tsit5()
    options = options or Options()

    drive_frequency = 2 * math.pi / drive_period
    save_times = jnp.linspace(0, drive_period, num_times)
    drive_photons = jnp.arange(-num_photons, num_photons + 1)

    quasienergies, modes = get_propagated_branches(
        hamiltonian=hamiltonian,
        drive_pulse=drive_pulse,
        drive_op=drive_op,
        drive_period=drive_period,
        save_times=save_times,
        method=method,
        options=options,
    )

    transition_frequencies = get_floquet_detunings(
        quasienergies=quasienergies,
        drive_photons=drive_photons,
        drive_frequency=drive_frequency,
    )

    floquet_matrix_elements = get_floquet_matrix_elements(
        modes=modes,
        times=save_times,
        drive_op=drive_op,
        drive_frequency=drive_frequency,
        drive_photons=drive_photons,
    )

    transition_rates = get_transition_rate(
        frequencies=transition_frequencies,
        spectral_density=spectral_density,
        thermal_populations=thermal_populations,
    )

    floquet_rates = transition_rates * jnp.abs(floquet_matrix_elements) ** 2
    mode_rates = jnp.sum(floquet_rates, -1)
    init_modes = jnp.take(modes, 0, -3)
    return quasienergies, init_modes, mode_rates
