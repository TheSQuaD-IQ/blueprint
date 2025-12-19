from typing import Callable

from jax import Array
from jax import jit
from jax import numpy as jnp
from jax.tree_util import Partial
from jax.typing import ArrayLike

from .envelope_library import eval_gaussian_env


@jit
def eval_cos_pulse(
    time: ArrayLike, amplitude: ArrayLike, frequency: ArrayLike, phase: ArrayLike
) -> Array:
    """
    eval_cos_pulse Evaluate a cosine pulse at the given time(s).

    Parameters
    ----------
    time : ArrayLike
        Time(s) at which to evaluate the pulse.
    amplitude : ArrayLike
        Pulse amplitude.
    frequency : ArrayLike
        Pulse frequency.
    phase : ArrayLike
        Pulse phase.

    Returns
    -------
    Array
        Value of the cosine pulse at ``time``.
    """
    pulse_val = amplitude * jnp.cos(frequency * time + phase)
    return pulse_val


@jit
def eval_sin_pulse(
    time: ArrayLike, amplitude: ArrayLike, frequency: ArrayLike, phase: ArrayLike
) -> Array:
    """
    eval_sin_pulse Evaluate a sine pulse at the given time(s).

    Parameters
    ----------
    time : ArrayLike
        Time(s) at which to evaluate the pulse.
    amplitude : ArrayLike
        Pulse amplitude.
    frequency : ArrayLike
        Pulse frequency.
    phase : ArrayLike
        Pulse phase.

    Returns
    -------
    Array
        Value of the sine pulse at ``time``.
    """
    pulse_val = amplitude * jnp.sin(frequency * time + phase)
    return pulse_val


def get_cos_pulse(
    amplitude: ArrayLike, frequency: ArrayLike, phase: ArrayLike
) -> Callable[[ArrayLike], Array]:
    """
    get_cos_pulse Return a cosine pulse function with fixed parameters.

    Parameters
    ----------
    amplitude : ArrayLike
        Pulse amplitude.
    frequency : ArrayLike
        Pulse frequency.
    phase : ArrayLike
        Pulse phase.

    Returns
    -------
    Callable[[ArrayLike], Array]
        A function that accepts ``time`` and returns the pulse value.
    """
    pulse = Partial(
        eval_cos_pulse, amplitude=amplitude, frequency=frequency, phase=phase
    )
    return pulse


def get_sin_pulse(
    amplitude: ArrayLike, frequency: ArrayLike, phase: ArrayLike
) -> Callable[[ArrayLike], Array]:
    """
    get_sin_pulse Return a sine pulse function with fixed parameters.

    Parameters
    ----------
    amplitude : ArrayLike
        Pulse amplitude.
    frequency : ArrayLike
        Pulse frequency.
    phase : ArrayLike
        Pulse phase.

    Returns
    -------
    Callable[[ArrayLike], Array]
        A function that accepts ``time`` and returns the pulse value.
    """
    pulse = Partial(
        eval_sin_pulse, amplitude=amplitude, frequency=frequency, phase=phase
    )
    return pulse


@jit
def eval_gaussian_cos_pulse(
    time: ArrayLike,
    amplitude: ArrayLike,
    frequency: ArrayLike,
    phase: ArrayLike,
    duration: ArrayLike,
    gaussian_std: ArrayLike,
) -> Array:
    """
    eval_gaussian_cos_pulse Evaluate a Gaussian-shaped cosine pulse at the given time(s).

    Parameters
    ----------
    time : ArrayLike
        Time(s) at which to evaluate the pulse.
    amplitude : ArrayLike
        Pulse amplitude.
    frequency : ArrayLike
        Pulse frequency.
    phase : ArrayLike
        Pulse phase.
    duration : ArrayLike
        Total duration of the pulse (used by the envelope evaluator).
    gaussian_std : ArrayLike
        Standard deviation of the Gaussian envelope.

    Returns
    -------
    Array
        Value of the Gaussian cosine pulse at ``time``.
    """
    envelope_val = eval_gaussian_env(time, duration, gaussian_std)
    pulse_val = eval_cos_pulse(time, amplitude, frequency, phase)
    return envelope_val * pulse_val


def get_gaussian_cos_pulse(
    amplitude: ArrayLike,
    frequency: ArrayLike,
    phase: ArrayLike,
    duration: ArrayLike,
    gaussian_std: ArrayLike,
) -> Callable[[ArrayLike], Array]:
    """
    get_gaussian_cos_pulse Return a Gaussian-shaped cosine pulse function with fixed parameters.

    Parameters
    ----------
    amplitude : ArrayLike
        Pulse amplitude.
    frequency : ArrayLike
        Pulse frequency.
    phase : ArrayLike
        Pulse phase.
    duration : ArrayLike
        Total duration of the pulse.
    gaussian_std : ArrayLike
        Standard deviation of the Gaussian envelope.

    Returns
    -------
    Callable[[ArrayLike], Array]
        A function that accepts ``time`` and returns the Gaussian cosine pulse.
    """
    pulse = Partial(
        eval_gaussian_cos_pulse,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        duration=duration,
        gaussian_std=gaussian_std,
    )
    return pulse
