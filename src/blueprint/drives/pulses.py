from typing import Callable

from jax import jit
from jax import numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Scalar

from .envelopes import eval_gaussian_env, eval_gaussian_drag_env


@jit
def eval_square_pulse(time: Array, amplitude: Scalar) -> Array:
    """
    square_pulse Evaluate a constant pulse at the given time(s).

    Parameters
    ----------
    time : Array
        Time(s) at which to evaluate the pulse.
    amplitude : Scalar
        Pulse amplitude.

    Returns
    -------
    Array
        Value of the constant pulse at ``time``.
    """
    pulse_val = amplitude * jnp.ones_like(time)
    return pulse_val


@jit
def eval_cos_pulse(
    time: Array, amplitude: Scalar, frequency: Scalar, phase: Scalar
) -> Array:
    """
    eval_cos_pulse Evaluate a cosine pulse at the given time(s).

    Parameters
    ----------
    time : Scalar
        Time(s) at which to evaluate the pulse.
    amplitude : Scalar
        Pulse amplitude.
    frequency : Scalar
        Pulse frequency.
    phase : Scalar
        Pulse phase.

    Returns
    -------
    Scalar
        Value of the cosine pulse at ``time``.
    """
    pulse_val = amplitude * jnp.cos(frequency * time + phase)
    return pulse_val


@jit
def eval_sin_pulse(
    time: Array, amplitude: Scalar, frequency: Scalar, phase: Scalar
) -> Array:
    """
    eval_sin_pulse Evaluate a sine pulse at the given time(s).

    Parameters
    ----------
    time : Scalar
        Time(s) at which to evaluate the pulse.
    amplitude : Scalar
        Pulse amplitude.
    frequency : Scalar
        Pulse frequency.
    phase : Scalar
        Pulse phase.

    Returns
    -------
    Scalar
        Value of the sine pulse at ``time``.
    """
    pulse_val = amplitude * jnp.sin(frequency * time + phase)
    return pulse_val


@jit
def eval_gaussian_cos_pulse(
    time: Scalar,
    amplitude: Scalar,
    frequency: Scalar,
    phase: Scalar,
    duration: Scalar,
    gaussian_std: Scalar,
    init_time: Scalar,
) -> Scalar:
    """
    eval_gaussian_cos_pulse Evaluate a Gaussian-shaped cosine pulse at the given time(s).

    Parameters
    ----------
    time : Scalar
        Time(s) at which to evaluate the pulse.
    amplitude : Scalar
        Pulse amplitude.
    frequency : Scalar
        Pulse frequency.
    phase : Scalar
        Pulse phase.
    duration : Scalar
        Total duration of the pulse (used by the envelope evaluator).
    gaussian_std : Scalar
        Standard deviation of the Gaussian envelope.

    Returns
    -------
    Scalar
        Value of the Gaussian cosine pulse at ``time``.
    """
    env = eval_gaussian_env(time, duration, gaussian_std, init_time)
    pulse = amplitude * env.real * jnp.cos(frequency * time + phase)
    return pulse


@jit
def eval_gaussian_sin_pulse(
    time: Scalar,
    amplitude: Scalar,
    frequency: Scalar,
    phase: Scalar,
    duration: Scalar,
    gaussian_std: Scalar,
    init_time: Scalar,
) -> Scalar:
    """
    eval_gaussian_sin_pulse Evaluate a Gaussian-shaped sine pulse at the given time(s).

    Parameters
    ----------
    time : Scalar
        Time(s) at which to evaluate the pulse.
    amplitude : Scalar
        Pulse amplitude.
    frequency : Scalar
        Pulse frequency.
    phase : Scalar
        Pulse phase.
    duration : Scalar
        Total duration of the pulse (used by the envelope evaluator).
    gaussian_std : Scalar
        Standard deviation of the Gaussian envelope.

    Returns
    -------
    Scalar
        Value of the Gaussian sine pulse at ``time``.
    """
    env = eval_gaussian_env(time, duration, gaussian_std, init_time)
    pulse = amplitude * env.real * jnp.sin(frequency * time + phase)
    return pulse


@jit
def eval_gaussian_drag_pulse(
    time: Scalar,
    amplitude: Scalar,
    frequency: Scalar,
    phase: Scalar,
    duration: Scalar,
    gaussian_std: Scalar,
    drag_coefficient: Scalar,
    init_time: Scalar,
) -> Scalar:
    """
    eval_gaussian_drag_pulse Evaluate a Gaussian DRAG pulse at the given time(s).

    Parameters
    ----------
    time : Scalar
        Time(s) at which to evaluate the pulse.
    amplitude : Scalar
        Pulse amplitude.
    frequency : Scalar
        Pulse frequency.
    phase : Scalar
        Pulse phase.
    duration : Scalar
        Total duration of the pulse (used by the envelope evaluator).
    gaussian_std : Scalar
        Standard deviation of the Gaussian envelope.
    drag_coefficient : Scalar
        DRAG coefficient.

    Returns
    -------
    Scalar
        Value of the Gaussian DRAG cosine pulse at ``time``.
    """
    env = eval_gaussian_drag_env(
        time, duration, gaussian_std, drag_coefficient, init_time
    )
    inphase_comp = env.real * jnp.cos(frequency * time + phase)
    quad_comp = env.imag * jnp.sin(frequency * time + phase)
    pulse = amplitude * (inphase_comp + quad_comp)
    return pulse


def get_square_pulse(amplitude: float | Scalar) -> Callable[[Scalar], Scalar]:
    """
    get_square_pulse Return a constant pulse function with fixed amplitude.

    Parameters
    ----------
    amplitude : float | Scalar
        Pulse amplitude.

    Returns
    -------
    Callable[[Scalar], Scalar]
        A function that accepts ``time`` and returns the pulse value.
    """
    amplitude = jnp.asarray(amplitude)

    pulse = Partial(
        eval_square_pulse,
        amplitude=amplitude,
    )
    return pulse


def get_cos_pulse(
    amplitude: float | Scalar, frequency: float | Scalar, phase: float | Scalar
) -> Callable[[Scalar], Scalar]:
    """
    get_cos_pulse Return a cosine pulse function with fixed parameters.

    Parameters
    ----------
    amplitude : float | Scalar
        Pulse amplitude.
    frequency : float | Scalar
        Pulse frequency.
    phase : float | Scalar
        Pulse phase.

    Returns
    -------
    Callable[[Scalar], Scalar]
        A function that accepts ``time`` and returns the pulse value.
    """
    amplitude = jnp.asarray(amplitude)
    frequency = jnp.asarray(frequency)
    phase = jnp.asarray(phase)

    pulse = Partial(
        eval_cos_pulse,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
    )
    return pulse


def get_sin_pulse(
    amplitude: float | Scalar, frequency: float | Scalar, phase: float | Scalar
) -> Callable[[Array], Scalar]:
    """
    get_sin_pulse Return a sine pulse function with fixed parameters.

    Parameters
    ----------
    amplitude : float | Scalar
        Pulse amplitude.
    frequency : float | Scalar
        Pulse frequency.
    phase : float | Scalar
        Pulse phase.

    Returns
    -------
    Callable[[Array], Scalar]
        A function that accepts ``time`` and returns the pulse value.
    """
    amplitude = jnp.asarray(amplitude)
    frequency = jnp.asarray(frequency)
    phase = jnp.asarray(phase)

    pulse = Partial(
        eval_sin_pulse,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
    )
    return pulse


def get_gaussian_cos_pulse(
    amplitude: float | Scalar,
    frequency: float | Scalar,
    phase: float | Scalar,
    duration: float | Scalar,
    gaussian_std: float | Scalar,
    init_time: float | Scalar = 0.0,
) -> Callable[[Array], Scalar]:
    """
    get_gaussian_cos_pulse Return a Gaussian-shaped cosine pulse function with fixed parameters.

    Parameters
    ----------
    amplitude : float | Scalar
        Pulse amplitude.
    frequency : float | Scalar
        Pulse frequency.
    phase : float | Scalar
        Pulse phase.
    duration : float | Scalar
        Total duration of the pulse.
    gaussian_std : float | Scalar
        Standard deviation of the Gaussian envelope.
    init_time : float | Scalar, optional
        Initial time of the pulse, by default 0.0

    Returns
    -------
    Callable[[Array], Scalar]
        A function that accepts ``time`` and returns the Gaussian cosine pulse.
    """
    amplitude = jnp.asarray(amplitude)
    frequency = jnp.asarray(frequency)
    phase = jnp.asarray(phase)
    duration = jnp.asarray(duration)
    gaussian_std = jnp.asarray(gaussian_std)

    pulse = Partial(
        eval_gaussian_cos_pulse,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        duration=duration,
        gaussian_std=gaussian_std,
        init_time=init_time,
    )
    return pulse


def get_gaussian_sin_pulse(
    amplitude: float | Scalar,
    frequency: float | Scalar,
    phase: float | Scalar,
    duration: float | Scalar,
    gaussian_std: float | Scalar,
    init_time: float | Scalar = 0.0,
) -> Callable[[Array], Scalar]:
    """
    get_gaussian_sin_pulse Return a Gaussian-shaped sine pulse function with fixed parameters.

    Parameters
    ----------
    amplitude : float | Scalar
        Pulse amplitude.
    frequency : float | Scalar
        Pulse frequency.
    phase : float | Scalar
        Pulse phase.
    duration : float | Scalar
        Total duration of the pulse.
    gaussian_std : float | Scalar
        Standard deviation of the Gaussian envelope.
    init_time : float | Scalar, optional
        Initial time of the pulse, by default 0.0

    Returns
    -------
    Callable[[Array], Scalar]
        A function that accepts ``time`` and returns the Gaussian sine pulse.
    """
    amplitude = jnp.asarray(amplitude)
    frequency = jnp.asarray(frequency)
    phase = jnp.asarray(phase)
    duration = jnp.asarray(duration)
    gaussian_std = jnp.asarray(gaussian_std)

    pulse = Partial(
        eval_gaussian_sin_pulse,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        duration=duration,
        gaussian_std=gaussian_std,
        init_time=init_time,
    )
    return pulse


def get_gaussian_drag_pulse(
    amplitude: float | Scalar,
    frequency: float | Scalar,
    phase: float | Scalar,
    duration: float | Scalar,
    gaussian_std: float | Scalar,
    drag_coefficient: float | Scalar,
    init_time: float | Scalar = 0.0,
) -> Callable[[Array], Scalar]:
    """
    get_gaussian_drag_pulse Return a Gaussian DRAG pulse function with fixed parameters.

    Parameters
    ----------
    amplitude : float | Scalar
        Pulse amplitude.
    frequency : float | Scalar
        Pulse frequency.
    phase : float | Scalar
        Pulse phase.
    duration : float | Scalar
        Total duration of the pulse.
    gaussian_std : float | Scalar
        Standard deviation of the Gaussian envelope.
    drag_coefficient : float | Scalar
        DRAG coefficient.
    init_time : float | Scalar, optional
        Initial time of the pulse, by default 0.0

    Returns
    -------
    Callable[[Array], Scalar]
        A function that accepts ``time`` and returns the Gaussian DRAG pulse.
    """
    amplitude = jnp.asarray(amplitude)
    frequency = jnp.asarray(frequency)
    phase = jnp.asarray(phase)
    duration = jnp.asarray(duration)
    gaussian_std = jnp.asarray(gaussian_std)
    drag_coefficient = jnp.asarray(drag_coefficient)
    init_time = jnp.asarray(init_time)

    pulse = Partial(
        eval_gaussian_drag_pulse,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        duration=duration,
        gaussian_std=gaussian_std,
        drag_coefficient=drag_coefficient,
        init_time=init_time,
    )
    return pulse
