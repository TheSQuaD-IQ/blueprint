from typing import Callable

from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp
from jax.typing import ArrayLike
from jax.tree_util import Partial


def eval_gaussian_env(
    time: ArrayLike, duration: ArrayLike, gaussian_std: ArrayLike
) -> Array:
    """
    eval_gaussian_env Evaluates a Gaussian envelope function at a given time.

    Parameters
    ----------
    time : ArrayLike
        The time at which to evaluate the envelope.
    duration : ArrayLike
        The duration of the envelope
    gaussian_std : ArrayLike
        The standard deviation of the Gaussian envelope.

    Returns
    -------
    Array
        The value of the Gaussian envelope at the given time.
    """
    half_duration = 0.5 * duration
    gaussian_term = jnp.exp(-0.5 * ((time - half_duration) / gaussian_std) ** 2)
    offset_term = jnp.exp(-0.5 * (half_duration / gaussian_std) ** 2)
    # envelope = (gaussian_term - offset_term) / (1 - offset_term)

    erf_term = jsp.special.erf(half_duration / (jnp.sqrt(2) * gaussian_std))

    denominator = (
        jnp.sqrt(2 * jnp.pi) * gaussian_std * erf_term - duration * offset_term
    )
    envelope = (gaussian_term - offset_term) / denominator

    return envelope


def get_gaussian_env(
    duration: ArrayLike, gaussian_std: ArrayLike
) -> Callable[[ArrayLike], Array]:
    """
    get_gaussian_env Returns a Gaussian envelope function with fixed

    Parameters
    ----------
    duration : ArrayLike
        The duration of the envelope.
    gaussian_std : ArrayLike
        The standard deviation of the Gaussian envelope.

    Returns
    -------
    Array
        The Gaussian envelope function with fixed parameters.
    """
    envelope = Partial(eval_gaussian_env, duration=duration, gaussian_std=gaussian_std)
    return envelope
