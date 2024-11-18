from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erf

from .envelopes import PulseParamType, format_pulse_params

__all__ = ["prepare_gaussian_params", "gaussian_filter_closure_func"]


def prepare_gaussian_params(
    pixel_times: Array, pixel_amplitudes: PulseParamType, gaussian_std: PulseParamType
) -> tuple[Array, Array]:
    r"""Returns the formatted parameters for the Gaussian filter.

    Since the Gaussian filter is taking a derivative in the discrete time dimension `p`,
    the `pixel_times` and `pixel_sizes` need to be of shape `p+1`. This function self-
    consistently does this expansion. Additionally, it adds a dimension `s` for batching
    over the Gaussian widths.

    Args:
        pixel_times  _(array of shape (p))_: Discretized times from which the pulse
            amplitudes are defined. `p` is the number of pixels.
        pixel_amplitudes _(array of shape (p, ...))_: Discretized pulse amplitudes
            used by the Gaussian filter. Can have arbitrary batching dimensions after
            the first time dimension `p`.
        gaussian_std _(array of shape (s))_: Gaussian filter standard deviations.
            `s` is the number of different gaussian filter widths to use.

    Returns:
        tuple[_(arrays of shape (p, 1), (1, s))_]: Formatted Gaussian filter
            parameters to use with `gaussian_filter_closure_func()`.
    """
    pixel_sizes = jnp.diff(pixel_times)
    pixel_sizes = jnp.concatenate([pixel_sizes, pixel_sizes[-1:]], axis=0)
    if len(pixel_times) == len(jnp.atleast_1d(pixel_amplitudes)):
        pixel_times = jnp.concatenate(
            [pixel_times, pixel_times[-1:] + pixel_sizes[-1]], axis=0
        )
        pixel_sizes = jnp.concatenate([pixel_sizes, pixel_sizes[-1:]], axis=0)
    mid_pixel_times = pixel_times - pixel_sizes / 2
    timescale = 1 / (jnp.sqrt(2) * jnp.atleast_1d(gaussian_std))[None]  # (1, Nsig)
    return mid_pixel_times[:, None], timescale


def gaussian_filter_closure_func(
    t: float, mid_pixel_times: Array, pixel_amplitudes: Array, timescale: Array
) -> Array:
    r"""Returns the Gaussian filtered pulse amplitudes at time `t`.

    Evaluating the convolution integral between an arbitrary signal and a Gaussian
    filter `$F(\omega) = \exp{(-\omega^2/\omega_0^2)}$`, where $\omega_0$ is the
    reference frequency of the filter, one obtains that the transfer matrix is given
    by a difference of `erf` functions as follow
    [Adapted from Eq. (5.11) of https://arxiv.org/abs/1102.0584]

    $T(t) = \frac{1}{2} \left(
        \erf{\omega_0 (\frac{t - pixel_times[:-1]}{2})} -
        \erf{\omega_0 (\frac{t - pixel_times[1:]}{2})}
    \right)$.

    Args:
        t (float): Time at which to evaluate the pulse amplitdes.
        mid_pixel_times _(array of shape (p+1))_: Discretized times from which the pulse
            amplitudes are defined. `p` is the number of pixels.
        pixel_amplitudes _(array of shape (p, ...))_: Discretized pulse amplitudes
            used by the Gaussian filter. Can have arbitrary batching dimensions after
            the first time dimension `p`.
        timescale _(array of shape (s))_: Normalized sizes of the gaussian filter
            standard deviations. Corresponds to $1 / (\sqrt{2}\sigma)$.
            `s` is the number of different gaussian filter widths to use.

    Returns:
        _(array of shape (s, ...))_ Gaussian filtered pulse amplitudes, with the filter
            width dimension `s` first, followed by all other potential batching dims.
    """
    erfs = -0.5 * jnp.diff(erf((t - mid_pixel_times) * timescale), axis=0)
    # erfs = erfs / jnp.sum(erfs, axis=0)
    output_amps = jnp.einsum("ps,p...->s...", erfs, pixel_amplitudes)
    return jnp.squeeze(output_amps)


def prepare_gaussian_IQmixer_params(
    pixel_times: Array,
    pixel_amplitudes: PulseParamType,
    gaussian_std: PulseParamType,
    LO_frequencies: PulseParamType,
    LO_phases: PulseParamType,
) -> tuple[Array, Array, Array, Array]:
    mid_pixel_times, timescale = prepare_gaussian_params(
        pixel_times, pixel_amplitudes, gaussian_std
    )
    LO_frequencies, LO_phases = format_pulse_params([LO_frequencies, LO_phases])
    return (
        mid_pixel_times.reshape(*mid_pixel_times.shape, 1, 1),
        timescale.reshape(*timescale.shape, 1, 1),
        LO_frequencies.reshape(1, 1, *LO_frequencies.shape),
        LO_phases.reshape(1, 1, *LO_phases.shape),
    )


def gaussian_filter_and_IQmixer_closure_func(
    t: float,
    mid_pixel_times: Array,
    pixel_amplitudes: Array,
    timescale: Array,
    LO_frequencies: Array,
    LO_phases: Array,
) -> Array:
    """_summary_

    Last dim of pixel_amplitudes needs to be the 2 quadrature for I and Q.

    Args:
        t (float): _description_
        mid_pixel_times (Array): _description_
        pixel_amplitudes (Array): _description_
        timescale (Array): _description_
        LO_frequencies (Array): _description_
        LO_phases (Array): _description_

    Returns:
        Array: _description_
    """
    # First, filter the envelope, without the carrier.
    erfs = -0.5 * jnp.diff(erf((t - mid_pixel_times) * timescale), axis=0)
    output_amps = jnp.einsum("psfc,p...->sfc...", erfs, pixel_amplitudes)
    # Then, IQ-mix with the LO to upconvert the signal to the desired frequency.
    return jnp.squeeze(
        output_amps[..., 0] * jnp.cos(LO_frequencies * t + LO_phases)
        + output_amps[..., 1] * jnp.sin(LO_frequencies * t + LO_phases)
    )
