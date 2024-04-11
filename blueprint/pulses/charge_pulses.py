import math
from jax import numpy as jnp
from jax import Array


def cosine_pulse(
    t: float,
    amplitude_GHz: float,
    gate_time_ns: float,
    carrier_freq_GHz: float,
    carrier_phase: float,
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
    carrier_phase: float,
) -> Array:
    """
    cosine_drag_pulse A simple cosine envelope with "quadrature" DRAG pulse that starts
    and end at 0 at t=0.0 and t=gate_time_ns, and reaches a maximal amplitude of
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


def gaussian_pulse(
    t: float,
    amplitude_GHz: float,
    gaussian_std_ns: float,
    number_of_stds: int,
    carrier_freq_GHz: float,
    carrier_phase: float,
) -> Array:
    """
    gaussian_pulse A simple Gaussian envelope that starts and end at 0 at t=0.0
    and t=gate_time_ns, and reaches a maximal amplitude of `amplitude_GHz`.

    Note: The amplitude to implement a theta rotation should be
    amplitude = theta / sqrt(2 * pi * sigma**2) / (
            erf(T/sqrt(8*sigma**2)) -
            T * exp(-T**2 / (8 * sigma**2))
        )
    according to Eq (3.2) of https://doi.org/10.1103/PhysRevA.83.012308.

    Args:
        t (float): Time, in ns.
        amplitude_GHz (float): _description_
        gate_time_ns (float): _description_
        carrier_freq_GHz (float): _description_
        carrier_phase (float, optional): _description_.

    Returns:
        float: _description_
    """
    gate_time_ns = gaussian_std_ns * number_of_stds
    half_t = gate_time_ns / 2
    return (
        amplitude_GHz
        * (
            jnp.exp(-0.5 * (t - half_t) ** 2 / gaussian_std_ns**2)
            - jnp.exp(-0.5 * half_t**2 / gaussian_std_ns**2)
        )
        * jnp.cos(carrier_freq_GHz * t + carrier_phase)
    )


def gaussian_drag_pulse(
    t: float,
    amplitude_GHz: float,
    gaussian_std_ns: float,
    number_of_stds: int,
    carrier_freq_GHz: float,
    carrier_phase: float,
    drag_param: float,
) -> Array:
    """
    gaussian_pulse A simple Gaussian envelope that starts and end at 0 at t=0.0
    and t=gate_time_ns, and reaches a maximal amplitude of `amplitude_GHz`.

    Note: The amplitude to implement a theta rotation should be
        amplitude = theta / sqrt(2 * pi * sigma**2) / (
                erf(T/sqrt(8*sigma**2)) -
                T * exp(-T**2 / (8 * sigma**2))
            )
        according to Eq (3.2) of https://doi.org/10.1103/PhysRevA.83.012308.

    Note2: for detuning=0, drag_param should be `-1/4/anharmonicity` according to
        Eq (4.34) of the same paper (Gambetta et al, 2011).

    Args:
        t (float): Time, in ns.
        amplitude_GHz (float): _description_
        gate_time_ns (float): _description_
        carrier_freq_GHz (float): _description_
        carrier_phase (float, optional): _description_.

    Returns:
        float: _description_
    """
    gate_time_ns = gaussian_std_ns * number_of_stds
    half_t = gate_time_ns / 2
    return gaussian_pulse(
        t,
        amplitude_GHz,
        gaussian_std_ns,
        number_of_stds,
        carrier_freq_GHz,
        carrier_phase,
    ) * (1 - drag_param * (t - half_t) / gaussian_std_ns)
