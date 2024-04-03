from typing import Union, Callable
import math
from jax import numpy as jnp
from jax.scipy.special import erf

from ..qubits.transmon import TunableTransmon


Numeric = Union[float, complex]


def gaussian_square(
    t: float,
    amplitude: Numeric,
    hold_time: float,
    ramp_time: float,
    std: float = 1.0,
) -> Numeric:
    """
    gaussian_square Returns a Gaussian squared pulse.

    Parameters
    ----------
    time : float
        The time at which to evaluate the pulse.
    amplitude: float
        The maximum amplitude of the pulse. This can refer to the strength of either a drive or a coupling.
    hold_time : float
        The duration that the pulse spends at the maximal amplitude.
    ramp_time : float
        The duration of the ramp up and ramp down of the pulse. These are assumed to be symmetric.
    std : float, optional
        The standard deviation of the Gaussian used to shape the rise/fall times, by default 1.0

    Returns
    -------
    float or complex
        The value of the pulse amplitude at time `time`.
    """
    if t < ramp_time:
        envelope = math.exp(-0.5 * (t - ramp_time) ** 2 / (std**2))
        return amplitude * envelope

    if t < ramp_time + hold_time:
        return amplitude

    if t < 2 * ramp_time + hold_time:
        drop_time = ramp_time + hold_time
        envelope = math.exp(-0.5 * (t - drop_time) ** 2 / (std**2))
        return amplitude * envelope

    return 0.0


def flat_top_gaussian(
    t: float,
    amplitude: Numeric,
    hold_time: float,
    buffer_start: float,
    buffer_end: float,
    gaussian_filter_sigma: float,
) -> Numeric:
    """
    flat_top_gaussian The rise and fall are determined by the gaussian filter std.

    Args:
        t (float): _description_
        amplitude (Numeric): _description_
        hold_time (float): _description_
        buffer_start (float): _description_
        buffer_end (float): _description_
        gaussian_filter_sigma (float): _description_

    Returns:
        Numeric: _description_
    """
    lengths = jnp.array([buffer_start, hold_time, buffer_end])
    amplitudes = jnp.array([0.0, amplitude, 0.0])
    timescale = 1 / (math.sqrt(2) * gaussian_filter_sigma)
    times_drives = jnp.concatenate(
        (
            jnp.zeros(
                1,
            ),
            jnp.cumsum(lengths),
        )
    )
    erfs = -0.5 * jnp.diff(erf((t - times_drives) * timescale), axis=0)
    pulse_amplitude = (erfs * amplitudes).sum(axis=0)
    return float(pulse_amplitude)


def capacitive_coupling_pulse(
    t: float,
    # transmon_1: TunableTransmon,
    # transmon_2: TunableTransmon,
    # flux_drive_transmon_1: callable,
    # flux_drive_transmon_2: callable,
    flux_drive_transmon_1: Callable,
    flux_drive_transmon_2: Callable,
    EJ_1: float,
    EJ_2: float,
    EC_1: float,
    EC_2: float,
    asymm_1: float,
    asymm_2: float,
    static_ext_flux_1: float,
    static_ext_flux_2: float,
    frequency_resonator: float,
    coupling_res_trans_1: float,
    coupling_res_trans_2: float,
) -> float:
    """
    capacitive_coupling_pulse

    J = g_1c g_2c / 2 * (1 / Delta_1c + 1 / Delta_2c)
    Eq(140) of RMP.
    """
    # Compute the external flux from the flux_drive callable
    total_external_flux_1 = static_ext_flux_1 + flux_drive_transmon_1(t)
    # Do the `compute_eff_josephson_energy()` part
    cos_term = math.cos(total_external_flux_1)
    sqrt_term = math.sqrt(1 + asymm_1**2 * math.tan(total_external_flux_1) ** 2)
    EJ_effective_1 = EJ_1 * abs(cos_term) * sqrt_term
    # Do the `ext_flux_to_approx_freq()` part
    freq_trans_1 = math.sqrt(8 * EC_1 * EJ_effective_1) - EC_1

    # Compute the external flux from the flux_drive callable
    total_external_flux_2 = static_ext_flux_2 + flux_drive_transmon_2(t)
    # Do the `compute_eff_josephson_energy()` part
    cos_term = math.cos(total_external_flux_2)
    sqrt_term = math.sqrt(1 + asymm_2**2 * math.tan(total_external_flux_2) ** 2)
    EJ_effective_2 = EJ_2 * abs(cos_term) * sqrt_term
    # Do the `ext_flux_to_approx_freq()` part
    freq_trans_2 = math.sqrt(8 * EC_2 * EJ_effective_2) - EC_2

    delta_res_trans_1 = freq_trans_1 - frequency_resonator
    delta_res_trans_2 = freq_trans_2 - frequency_resonator
    return (
        coupling_res_trans_1
        * coupling_res_trans_2
        / 2
        * (1 / delta_res_trans_1 + 1 / delta_res_trans_2)
    )
