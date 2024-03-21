from typing import Union
from math import exp

Numeric = Union[float, complex] 

def gaussian_square(
    time: float,
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
    if time < ramp_time:
        envelope = exp(-0.5 * (time - ramp_time) ** 2 / (std**2))
        return amplitude * envelope

    if time < ramp_time + hold_time:
        return amplitude

    if time < 2 * ramp_time + hold_time:
        drop_time = ramp_time + hold_time
        envelope = exp(-0.5 * (time - drop_time) ** 2 / (std**2))
        return amplitude * envelope

    return 0.0
