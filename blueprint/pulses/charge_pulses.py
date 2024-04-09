import math
from jax import numpy as jnp
from jax import Array


def cosine_pulse(
    t: float,
    amplitude_GHz: float,
    gate_time_ns: float,
    carrier_freq_GHz: float,
    carrier_phase: float = 0.0,
) -> Array:
    """
    cosine_pulse A simple cosine envelope that starts and end at 0 amplitude at t=0.0
    and t=gate_time_ns, and reaches a maximal amplitude of `amplitude_GHz`.

    Note: The mean amplitude is `amplitude_GHz/2` such that to implement a rotation of
        angle theta, the prefactor in front of sigma_j/2 in the Hamiltonian should be
        `amplitude = theta / gate_time_ns` and `amplitude_GHz = amplitude / (2 * pi)`.

    Args:
        t (float): Time, in ns.
        amplitude_GHz (float): _description_
        gate_time_ns (float): _description_
        carrier_freq_GHz (float): _description_
        carrier_phase (float, optional): _description_. Defaults to 0.0.

    Returns:
        float: _description_
    """
    return (
        amplitude_GHz
        * (1 - jnp.cos(2 * jnp.pi / gate_time_ns * t))
        / 2
        * jnp.cos(carrier_freq_GHz * t + carrier_phase)
    )


def cosine_drag_pulse(
    t: float,
    amplitude_GHz: float,
    gate_time_ns: float,
    carrier_freq_GHz: float,
    drag_param: float,
    carrier_phase: float = 0.0,
) -> Array:
    """
    cosine_drag_pulse A simple cosine envelope with in-phase DRAG pulse that starts and
    end at 0 amplitude at t=0.0 and t=gate_time_ns, and reaches a maximal amplitude of
    `amplitude_GHz`.

    Note: The mean amplitude is `amplitude_GHz/2` such that to implement a rotation of
        angle theta, the prefactor in front of sigma_j/2 in the Hamiltonian should be
        `amplitude = theta / gate_time_ns` and `amplitude_GHz = amplitude / (2 * pi)`.

    Args:
        t (float): Time, in ns.
        amplitude_GHz (float): _description_
        gate_time_ns (float): _description_
        carrier_freq_GHz (float): _description_
        carrier_phase (float, optional): _description_. Defaults to 0.0.

    Returns:
        float: _description_
    """
    return cosine_pulse(
        t, amplitude_GHz, gate_time_ns, carrier_freq_GHz, carrier_phase
    ) + (
        amplitude_GHz
        * drag_param
        * jnp.sin(2 * jnp.pi / gate_time_ns * t)
        * (jnp.pi / gate_time_ns)
        * jnp.sin(carrier_freq_GHz * t + carrier_phase)
    )
