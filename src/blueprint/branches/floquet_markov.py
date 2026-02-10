"""Floquet-Markov rate utilities.

This module provides helpers to compute Floquet quasienergies, matrix
elements and transition rates using JAX. Functions document expected
semantics but not full shapes; callers should ensure inputs are
JAX-compatible arrays with appropriate dtypes (real/complex).
"""

from jax import jit
from jax import numpy as jnp
from jaxtyping import Array, Scalar

from dynamiqs import Options
from dynamiqs.method import Method, Tsit5

from ..drives import Pulse
from .floquet import get_branches

type Float = float | Scalar


@jit
def get_floquet_detunings(
    energies: Array,
    photons: Array,
    drive_frequency: Float,
) -> Array:
    """
    get_floquet_detunings Calculates the detunings for transitions between Floquet modes that include the absorption or emission of a given number of drive photons.

    Parameters
    ----------
    energies : Array
        The quasienergies of the Floquet modes, with shape ``(num_amplitude, num_modes,)``.
    photons : Array
        The numbers of drive photons that can be absorbed or emitted, with shape ``(num_photons,)``.
    drive_frequency : Float
        The frequency of the drive.

    Returns
    -------
    Array
        The detunings for transitions between Floquet modes, with shape ``(num_amplitude, num_modes, num_modes, num_photons)``.
    """
    row_energies = jnp.expand_dims(energies, (-2, -1))
    col_energies = jnp.expand_dims(energies, (-3, -1))

    floquet_detunings = row_energies - col_energies + drive_frequency * photons
    return floquet_detunings


@jit
def get_floquet_mat_elements(
    modes: Array,
    times: Array,
    drive_op: Array,
    drive_frequency: Float,
    drive_photons: Array,
) -> Array:
    """
    get_floquet_mat_elements Computes the Floquet matrix elements for transitions between Floquet modes.

    Parameters
    ----------
    modes : Array
        The Floquet modes of the system, with shape ``(num_amplitudes, num_times, dim, num_modes)``.
    times : Array
        The time points at which the Floquet modes are evaluated, with shape ``(num_times,)``.
    drive_op : Array
        The drive operator, with shape ``(dim, dim)``.
    drive_frequency : Float
        The frequency of the drive.
    drive_photons : Array
        The numbers of drive photons that can be absorbed or emitted, with shape ``(num_photons,)``.

    Returns
    -------
    Array
        The Floquet matrix elements, with shape ``(num_amplitudes, num_modes, num_modes, num_photons)``.
    """
    phases = jnp.exp(-1.0j * drive_frequency * jnp.outer(times, drive_photons))

    mat_elements = jnp.einsum(
        "atik, ij, atjl, tn -> atkln",
        jnp.conj(modes),
        drive_op,
        modes,
        phases,
        optimize=True,
    )

    drive_period = 2 * jnp.pi / drive_frequency
    num_times = jnp.size(times)

    time_step = drive_period / num_times

    floquet_mat_elements = (
        jnp.trapezoid(mat_elements, dx=time_step, axis=1) / drive_period
    )
    return floquet_mat_elements


@jit
def get_transition_rate(
    frequencies: Array, spectral_density: Array, thermal_populations: Array
) -> Array:
    pos_filter = jnp.heaviside(frequencies, 1.0)
    absorption_rates = thermal_populations * spectral_density * pos_filter
    emission_rates = (1 + thermal_populations) * spectral_density * (1 - pos_filter)
    transition_rates = absorption_rates + emission_rates
    return transition_rates


def get_branch_rates(
    hamiltonian: Array,
    drive_op: Array,
    drive_pulse: Pulse,
    drive_period: Float,
    spectral_density: Array,
    thermal_populations: Array,
    num_photons: int = 4,
    num_times: int = 1000,
    method: Method | None = None,
    options: Options | None = None,
) -> tuple[Array, Array, Array]:
    method = method or Tsit5()
    options = options or Options()

    times = jnp.linspace(0, drive_period, num_times)
    drive_photons = jnp.arange(-num_photons, num_photons + 1)
    drive_frequency = 2 * jnp.pi / drive_period

    quasienergies, modes = get_branches(
        hamiltonian=hamiltonian,
        drive_pulse=drive_pulse,
        drive_op=drive_op,
        drive_period=drive_period,
        time=times,
        method=method,
        options=options,
    )

    transition_frequencies = get_floquet_detunings(
        quasienergies=quasienergies,
        drive_photons=drive_photons,
        drive_frequency=drive_frequency,
    )

    mat_elements = get_floquet_mat_elements(
        modes=modes,
        times=times,
        drive_op=drive_op,
        drive_frequency=drive_frequency,
        drive_photons=drive_photons,
    )

    rates = get_transition_rate(
        frequencies=transition_frequencies,
        spectral_density=spectral_density,
        thermal_populations=thermal_populations,
    )

    mode_rates = jnp.sum(rates * jnp.abs(mat_elements) ** 2, -1)
    return quasienergies, modes, mode_rates
