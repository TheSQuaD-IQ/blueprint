"""Floquet-Markov rate utilities.

This module provides helpers to compute Floquet quasienergies, matrix
elements and transition rates using JAX. Functions document expected
semantics but not full shapes; callers should ensure inputs are
JAX-compatible arrays with appropriate dtypes (real/complex).
"""

from jax import jit
from jax import numpy as jnp
from jaxtyping import Array, Scalar

type Float = float | Scalar


@jit
def get_detunings(
    energies: Array,
    drive_photons: Array,
    drive_frequency: Float,
) -> Array:
    """
    get_detunings Calculates the detunings for transitions between Floquet modes that include the absorption or emission of a given number of drive photons.

    Parameters
    ----------
    energies : Array
        The quasienergies of the Floquet modes, with shape ``(num_amplitude, num_modes,)``.
    drive_photons : Array
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

    floquet_detunings = row_energies - col_energies + drive_frequency * drive_photons
    return floquet_detunings


@jit
def get_matrix_elements(
    modes: Array,
    times: Array,
    operator: Array,
    drive_frequency: Float,
    drive_photons: Array,
) -> Array:
    """
    get_matrix_elements Computes the Floquet matrix elements for transitions between Floquet modes.

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
        operator,
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
def get_density(
    frequency: Array, spectral_density: Array, thermal_population: Array
) -> Array:
    pos_filter = jnp.heaviside(frequency, 1.0)
    absorption_rates = thermal_population * spectral_density * pos_filter
    emission_rates = (1 + thermal_population) * spectral_density * (1 - pos_filter)
    transition_rates = absorption_rates + emission_rates
    return transition_rates


@jit
def get_transition_rates(
    frequencies: Array,
    matrix_elements: Array,
    spectral_density: Array,
    thermal_population: Array,
) -> Array:

    densities = get_density(
        frequency=frequencies,
        spectral_density=spectral_density,
        thermal_population=thermal_population,
    )

    transition_rates = jnp.sum(densities * jnp.abs(matrix_elements) ** 2, -1)
    return transition_rates
