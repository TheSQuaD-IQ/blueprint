import jax
from jax import jit
from jax import numpy as jnp
from jaxtyping import Array, Scalar


@jit
def eval_raised_cos_env(
    time: Array,
    duration: Scalar,
    init_time: Scalar,
) -> Array:
    """
    eval_raised_cos_env Evaluate a raised cosine envelope at the given time(s).

    Parameters
    ----------
    time : Array
        Time(s) at which to evaluate the envelope.
    duration : Scalar
        Total duration for the envelope.
    init_time : Scalar
        Initial time of the pulse.

    Returns
    -------
    Array
        The generated raised cosine waveform as a complex array.
    """
    # Unnormalized raised cosine
    cos = jnp.cos(2 * jnp.pi * time / duration)

    # Normalized raised cosine envelope
    env = 0.5 * (1.0 - cos)

    # Ensure the waveform is zero outside the pulse duration
    end_time = init_time + duration
    mask = (time >= init_time) & (time <= end_time)
    zeros = jnp.zeros_like(env)
    env = jax.lax.select(mask, env, zeros)
    return env


@jit
def eval_gaussian_env(
    time: Array,
    duration: Scalar,
    gaussian_std: Scalar,
    init_time: Scalar,
) -> Array:
    """
    eval_gaussian_env Evaluate a Gaussian envelope at the given time(s).

    Parameters
    ----------
    time : Array
        Time(s) at which to evaluate the envelope.
    duration : Scalar
        Total duration for the envelope.
    gaussian_std : Scalar
        Standard deviation of the Gaussian envelope.
    init_time : Scalar
        Initial time of the pulse.

    Returns
    -------
    Array
        The generated Gaussian waveform as a complex array.
    """
    half_duration = 0.5 * duration

    gaussian_mean = init_time + half_duration
    offset_time = time - gaussian_mean

    offset_term = jnp.exp(-0.5 * (half_duration / gaussian_std) ** 2)

    # Unnormalized Gaussian
    gaussian_term = jnp.exp(-0.5 * (offset_time / gaussian_std) ** 2)

    # Normalized Gaussian envelope
    env = (gaussian_term - offset_term) / (1.0 - offset_term)

    # erf_term = jsp.special.erf(half_duration / (jnp.sqrt(2) * gaussian_std))
    # denominator = jnp.sqrt(2 * jnp.pi) * gaussian_std * erf_term - duration * offset_term
    # env = (gaussian_term - offset_term) / denominator

    # Ensure the waveform is zero outside the pulse duration
    end_time = init_time + duration
    mask = (time >= init_time) & (time <= end_time)
    zeros = jnp.zeros_like(env)
    env = jax.lax.select(mask, env, zeros)
    return env


@jit
def eval_gaussian_drag_env(
    time: Array,
    duration: Scalar,
    gaussian_std: Scalar,
    drag_coefficient: Scalar,
    init_time: Scalar,
) -> Array:
    """
    eval_drag_gaussian_env Evaluate a DRAG Gaussian envelope at the given time(s).

    Parameters
    ----------
    time : Array
        Time(s) at which to evaluate the envelope.
    duration : Scalar
        Total duration for the envelope.
    gaussian_std : Scalar
        Standard deviation of the Gaussian envelope.
    drag_coefficient : Scalar
        DRAG coefficient for the derivative component.
    init_time : Scalar
        Initial time of the pulse
    Returns
    -------
    Array
        The generated DRAG Gaussian waveform as a complex array.

    """
    half_duration = 0.5 * duration

    gaussian_mean = init_time + half_duration
    offset_time = time - gaussian_mean

    offset_term = jnp.exp(-0.5 * (half_duration / gaussian_std) ** 2)

    # Unnormalized Gaussian
    gaussian_term = jnp.exp(-0.5 * (offset_time / gaussian_std) ** 2)

    # Normalized Gaussian envelope
    gaussian_env = (gaussian_term - offset_term) / (1.0 - offset_term)

    # Derivative of the normalized Gaussian:
    gaussian_deriv = -offset_time * gaussian_term / (gaussian_std**2)
    deriv_env = gaussian_deriv / (1.0 - offset_term)

    # DRAG combination
    env = gaussian_env + 1j * drag_coefficient * deriv_env

    # Mask outside pulse window
    end_time = init_time + duration
    mask = (time >= init_time) & (time <= end_time)
    zeros = jnp.zeros_like(env)
    env = jax.lax.select(mask, env, zeros)
    return env
