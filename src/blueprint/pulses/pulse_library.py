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
    eval_cos_pulse Evaluates the value of a cosine pulse at a given time.

    Parameters
    ----------
    time : ArrayLike
        The time at which to evaluate the pulse.
    frequency : ArrayLike
        The frequency of the pulse.
    phase : ArrayLike
        The phase of the pulse.

    Returns
    -------
    Array
        The value of the cosine pulse at the given time.
    """
    pulse_val = amplitude * jnp.cos(frequency * time + phase)
    return pulse_val


@jit
def eval_sin_pulse(
    time: ArrayLike, amplitude: ArrayLike, frequency: ArrayLike, phase: ArrayLike
) -> Array:
    """
    eval_sin_pulse Evaluates the value of a sine pulse at a given time.

    Parameters
    ----------
    time : ArrayLike
        The time at which to evaluate the pulse.
    frequency : ArrayLike
        The frequency of the pulse.
    phase : ArrayLike
        The phase of the pulse.

    Returns
    -------
    Array
        The value of the sine pulse at the given time.
    """
    pulse_val = amplitude * jnp.sin(frequency * time + phase)
    return pulse_val


def get_cos_pulse(
    amplitude: ArrayLike, frequency: ArrayLike, phase: ArrayLike
) -> Callable[[ArrayLike], Array]:
    """
    get_cos_pulse Returns a cosine pulse function with fixed
    pulse parameters that can be evaluated at each point in time.

    Parameters
    ----------
    frequency : ArrayLike
        The frequency of the pulse.
    phase : ArrayLike
        The phase of the pulse.

    Returns
    -------
    Callable[[ArrayLike], Array]
        The cosine pulse that can be evaluated at each time.
    """
    pulse = Partial(
        eval_cos_pulse, amplitude=amplitude, frequency=frequency, phase=phase
    )
    return pulse


def get_sin_pulse(
    amplitude: ArrayLike, frequency: ArrayLike, phase: ArrayLike
) -> Callable[[ArrayLike], Array]:
    """
    get_sin_pulse Returns a sine pulse function with fixed
    pulse parameters that can be evaluated at each point in time.

    Parameters
    ----------
    frequency : ArrayLike
        The frequency of the pulse.
    phase : ArrayLike
        The phase of the pulse.

    Returns
    -------
    Callable[[ArrayLike], Array]
        The sine pulse that can be evaluated at each time.
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
    eval_gaussian_cos_pulse Evaluates a Gaussian cosine pulse at a given time.

    Parameters
    ----------
    time : ArrayLike
        The time at which to evaluate the pulse.
    amplitude : ArrayLike
        The amplitude of the pulse.
    frequency : ArrayLike
        The frequency of the pulse.
    phase : ArrayLike
        The phase of the pulse.
    duration : ArrayLike
        The duration of the pulse.
    gaussian_std : ArrayLike
        The standard deviation of the Gaussian envelope.

    Returns
    -------
    Array
        The value of the Gaussian cosine pulse at the given time
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
    get_gaussian_cos_pulse Returns a Gaussian cosine pulse function with fixed pulse parameters
    that can be evaluated at each point in time.

    Parameters
    ----------
    amplitude : ArrayLike
        The amplitude of the pulse.
    frequency : ArrayLike
        The frequency of the pulse.
    phase : ArrayLike
        The phase of the pulse.
    duration : ArrayLike
        The duration of the pulse.
    gaussian_std : ArrayLike
        The standard deviation of the Gaussian envelope.

    Returns
    -------
    Callable[[ArrayLike], Array]
        The Gaussian cosine pulse that can be evaluated at each time.
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
