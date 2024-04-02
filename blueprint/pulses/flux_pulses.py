import math


def modulated_flux_pulse(
    t: float,
    max_voltage: float,
    flux_per_volt: float,
    modulation_freq: float,
    ramp_time: float,
    hold_time: float,
    std: float = 1.0,
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
