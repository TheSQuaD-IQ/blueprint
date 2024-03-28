import math

def modulated_flux_pulse(
    time: float,
    prop_factor: float,
    voltage_amp: float,
    modulation_freq: float,
    ramp_time: float,
    hold_time: float,
    std: float = 1.0,
) -> float:
    cos_term = math.cos(modulation_freq * time)
    sqrt2_std = math.sqrt(2) * std
    rise_term = math.erf((time - ramp_time) / sqrt2_std)
    fall_term = math.erf((time - ramp_time - hold_time) / sqrt2_std)
    input_voltage = 0.5 * voltage_amp * cos_term * (rise_term - fall_term)
    applied_flux = prop_factor * input_voltage / voltage_amp
    return applied_flux