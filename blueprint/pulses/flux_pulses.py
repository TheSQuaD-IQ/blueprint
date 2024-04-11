import math
from jax import numpy as jnp
from jax.scipy.special import erf
from jax import Array


def modulated_flux_pulse(
    t: float,
    max_voltage: float,
    flux_per_volt: float,
    modulation_freq: float,
    ramp_time: float,
    hold_time: float,
    std: float,
) -> float:
    """
    modulated_flux_pulse The pulse function for a modulated flux pulse. This pulse can be for example used to implement a LRU operation following the ETH paper TODO: add reference and exapand on the pulse.

    Parameters
    ----------
    t : float
        The time at which to evaluate the pulse.
    prop_factor : float
        The proportionality factor between the input voltage and the applied flux.
    voltage_amp : float
        The amplitude of the input voltage.
    modulation_freq : float
        The frequency of the modulation.
    ramp_time : float
        The time it takes for the pulse to ramp up.
    hold_time : float
        The time the pulse is held at the maximum value.
    std : float, optional
        The guassian standard deviation, used for the Gaussian ramp periods, by default 1.0

    Returns
    -------
    float
        The applied flux at time t.
    """
    cos_term = math.cos(modulation_freq * t)

    sqrt2_std = math.sqrt(2) * std
    rise_term = math.erf((t - ramp_time) / sqrt2_std)
    fall_term = math.erf((t - ramp_time - hold_time) / sqrt2_std)
    input_voltage = 0.5 * max_voltage * cos_term * (rise_term - fall_term)

    applied_flux = input_voltage * flux_per_volt
    return applied_flux


def net_zero_transition_flux_pulse(
    t: float,
    hold_first_voltage: float,
    transition_voltage: float,
    half_hold_time: float,
    transition_time: float,
    buffer_start: float,
    buffer_end: float,
    flux_per_volt: float,
    gaussian_filter_sigma: float,
) -> Array:
    """
    net_zero_transition_flux_pulse The Net Zero Transition flux pulse used for CPhase gates.

    Note: Voltages can also be fluxes directly if one uses `flux_per_volt = 1.0`.

    Note2: Make sure the units match such that `time / gaussian_filter_sigma` is unitless.

    Args:
        t (float): _description_
        hold_first_voltage (float): _description_
        transition_voltage (float): _description_
        half_hold_time (float): _description_
        transition_time (float): _description_
        buffer_start (float): _description_
        buffer_end (float): _description_
        flux_per_volt (float): _description_
        gaussian_filter_sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        float: _description_
    """
    lengths = jnp.array(
        [
            0.0,
            buffer_start,
            half_hold_time,
            transition_time,
            half_hold_time,
            buffer_end,
        ]
    )
    voltages = jnp.array(
        [0.0, hold_first_voltage, transition_voltage, -hold_first_voltage, 0.0]
    )
    timescale = 1 / (math.sqrt(2) * gaussian_filter_sigma)
    times_drives = jnp.cumsum(lengths)
    erfs = -0.5 * jnp.diff(erf((t - times_drives) * timescale), axis=0)
    erfs = erfs / jnp.sum(erfs)
    pulse_voltage = (erfs * voltages).sum(axis=0)
    applied_flux = pulse_voltage * flux_per_volt
    return applied_flux


def net_zero_transition_flux_pulse_batchable_voltage(
    t: float,
    hold_first_voltage: jnp.ndarray,
    transition_voltage: float,
    half_hold_time: float,
    transition_time: float,
    buffer_start: float,
    buffer_end: float,
    flux_per_volt: float,
    gaussian_filter_sigma: float,
) -> Array:
    """
    net_zero_transition_flux_pulse The Net Zero Transition flux pulse used for CPhase gates.

    Note: Voltages can also be fluxes directly if one uses `flux_per_volt = 1.0`.

    Note2: Make sure the units match such that `time / gaussian_filter_sigma` is unitless.

    Args:
        t (float): _description_
        hold_first_voltage (float): _description_
        transition_voltage (float): _description_
        half_hold_time (float): _description_
        transition_time (float): _description_
        buffer_start (float): _description_
        buffer_end (float): _description_
        flux_per_volt (float): _description_
        gaussian_filter_sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        float: _description_
    """
    lengths = jnp.array(
        [
            0.0,
            buffer_start,
            half_hold_time,
            transition_time,
            half_hold_time,
            buffer_end,
        ]
    )
    voltages = jnp.stack(
        [
            jnp.array([0.0, voltage, transition_voltage, -voltage, 0.0])
            for voltage in hold_first_voltage
        ]
    )
    timescale = 1 / (math.sqrt(2) * gaussian_filter_sigma)
    times_drives = jnp.cumsum(lengths)
    erfs = -0.5 * jnp.diff(erf((t - times_drives) * timescale), axis=0)
    erfs = erfs / jnp.sum(erfs)
    pulse_voltages = (erfs[None, :] * voltages).sum(axis=1)
    applied_fluxes = pulse_voltages * flux_per_volt
    return applied_fluxes
