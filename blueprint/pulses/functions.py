from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.tree_util import Partial
from typing import Callable

from .envelopes import (
    PulseParamType,
    _flat_envelope,
    _net_zero_envelope,
    _raised_cosine_envelope,
    _raised_cosine_drag_envelope_awg,
    format_pulse_params,
)
from .filters import (
    gaussian_filter_closure_func,
    prepare_gaussian_params,
    prepare_gaussian_IQmixer_params,
    gaussian_filter_and_IQmixer_closure_func,
)

__all__ = ["flat_top_gaussian", "raised_cosine", "raised_cosine_gaussian_filtered"]


def gaussian_filtered_func(
    pixel_times: Array, pixel_amplitudes: Array, gaussian_std: PulseParamType
) -> Callable[[float], Array]:
    """Returns the Gaussian filter function with signature `f(t: float) -> Array` by
    taking a partial over the constant pulse parameters. This decomposition is used to
    optimize the runtime of the pulse function used by the dynamiqs solvers.

    Args:
        pixel_times _(array of shape (p))_: Discretized times from which the pulse
            amplitudes are defined. `p` is the number of pixels.
        pixel_amplitudes _(array of shape (p, ...))_: Discretized pulse amplitudes
            used by the Gaussian filter. Can have arbitrary batching dimensions after
            the first time dimension `p`.
        gaussian_std _(array of shape (s))_: Gaussian filter standard deviations.
            `s` is the number of different gaussian filter widths to use.

    Returns:
        callable[[float], Array]: Function that returns the pulse amplitudes at an
            arbitrary time `t`. The array returned by this function has shape
            (gaussian_std?, *pixel_amplitudes.shape).
    """
    mid_pixel_times, timescale = prepare_gaussian_params(
        pixel_times=pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        gaussian_std=gaussian_std,
    )

    # Shape (batch_filter?, ...all_amplitudes_dim)
    return Partial(
        gaussian_filter_closure_func,
        mid_pixel_times=mid_pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        timescale=timescale,
    )


def flat_top_gaussian(
    pixel_times: Array,
    hold_amplitudes: PulseParamType,
    pad_times: PulseParamType,
    hold_times: PulseParamType,
    gaussian_std: PulseParamType,
) -> Callable[[float], Array]:
    """Returns a flat-top gaussian pulse function with signature `f(t: float) -> Array`,
    with nominal amplitudes `hold_amplitudes` during `hold_times` and 0.0 otherwise.
    The `pad_times` are applied both before and after the `hold_times`.

    Args:
        pixel_times _(array of shape (p))_: Discretized times from which the pulse
            amplitudes are defined. `p` is the number of pixels.
        hold_amplitudes (PulseParamType): Amplitudes of the pulse during the hold times.
        pad_times (PulseParamType): Zero paddings before and after the hold times.
        hold_times (PulseParamType): Durations of the flat pulse.
        gaussian_std _(array of shape (s))_: Gaussian filter standard deviations.
            `s` is the number of different gaussian filter widths to use.

    Returns:
        callable[[float], Array]: Flat-top gaussian pulse function. The array returned
            by this function has shape (gaussian_std?, hold_amplitudes?, pad_times?,
            hold_times?).
    """
    pixel_times_expanded, hold_amplitudes, pad_times, hold_times = format_pulse_params(
        [pixel_times, hold_amplitudes, pad_times, hold_times]
    )
    # Shape (pixel_dim, hold_amplitudes?, pad_times?, hold_times?)
    pixel_amplitudes = jnp.squeeze(
        hold_amplitudes
        * _flat_envelope(
            pixel_times_expanded, pad_times=pad_times, hold_times=hold_times
        )
    )

    # Shape (gaussian_std?, hold_amplitudes?, pad_times?, hold_times?)
    return gaussian_filtered_func(
        pixel_times=pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        gaussian_std=gaussian_std,
    )


def net_zero_gaussian_filtered(
    pixel_times: Array,
    hold_amplitudes: PulseParamType,
    pad_times: PulseParamType,
    hold_times: PulseParamType,
    transition_times: PulseParamType,
    gaussian_std: PulseParamType,
) -> Callable[[float], Array]:
    """Returns a net zero gaussian pulse function with signature `f(t: float) -> Array`,
    with nominal amplitudes `+1.0 * hold_amplitudes` during the first half of
    `hold_times`, `-1.0 * hold_amplitudes` during the second half, and `0.0` otherwise,
    i.e. during the initial and final `pad_times` and during the `transition_times` in
    between the two half hold times.

    Note:
        This nominal function pulse envelope is then Gaussian filtered.

    Pulse shape:
            2
         -------
        |       |
      1 |       | 3            1
    ----         ---         ----
                    |       |
                    |   2   |
                     -------

    1: `pad_times` with amplitude 0.0;
    2: Half of `hold_times` with amplitude +1* then -1* `hold_amplitudes`;
    3: `transition_times` with amplitude 0.0.

    Args:
        pixel_times _(array of shape (p))_: Discretized times from which the pulse
            amplitudes are defined. `p` is the number of pixels.
        hold_amplitudes (PulseParamType): Amplitudes of the pulse during the hold times.
        pad_times (PulseParamType): Zero paddings before and after the hold times.
        hold_times (PulseParamType): Durations of the flat pulse.
        gaussian_std _(array of shape (s))_: Gaussian filter standard deviations.
            `s` is the number of different gaussian filter widths to use.

    Returns:
        callable[[float], Array]: Flat-top gaussian pulse function. The array returned
            by this function has shape (gaussian_std?, hold_amplitudes?, pad_times?,
            hold_times?).
    """
    pixel_times_expanded, hold_amplitudes, pad_times, hold_times, transition_times = (
        format_pulse_params(
            [pixel_times, hold_amplitudes, pad_times, hold_times, transition_times]
        )
    )
    # Shape (pixel_dim, hold_amplitudes?, pad_times?, hold_times?)
    pixel_amplitudes = jnp.squeeze(
        hold_amplitudes
        * _net_zero_envelope(
            pixel_times_expanded,
            pad_times=pad_times,
            hold_times=hold_times,
            transition_times=transition_times,
        )
    )

    # Shape (gaussian_std?, hold_amplitudes?, pad_times?, hold_times?)
    return gaussian_filtered_func(
        pixel_times=pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        gaussian_std=gaussian_std,
    )


def raised_cosine(
    amplitudes: PulseParamType,
    gate_times: PulseParamType,
    carrier_freqs: PulseParamType,
    carrier_phases: PulseParamType,
) -> Callable[[float], Array]:
    """Returns a raised cosine pulse function with signature `f(t: float) -> Array`,
    which starts and end at 0 amplitude at `t=0.0` and `t=gate_time`, and reaches a
    maximal amplitude of `amplitudes` at `gate_time/2`.

    Args:
        amplitudes (PulseParamType): Maximal amplitudes of the pulses.
        gate_times (PulseParamType): Total durations of the gate.
        carrier_freqs (PulseParamType): Carrier frequencies of the pulse, i.e. the
            frequencies of the fast oscillations inside the smooth cosine envelope.
            Defaults to 0.0 which corresponds to getting only the smooth envelope.
        carrier_phases (PulseParamType): Phases of the carrier at t=0.0.

    Returns:
        callable[[float], Array]: Raised cosine pulse function. The array returned
            by this function has shape (amplitudes?, gate_times?, carrier_freqs?,
            carrier_phases?).
    """
    amplitudes, gate_times, carrier_freqs, carrier_phases = format_pulse_params(
        (amplitudes, gate_times, carrier_freqs, carrier_phases)
    )
    # Shape (Npix, gate_times?, carrier_freqs?, carrier_phases?)

    def closure_func(
        t: float,
        amplitudes: Array,
        gate_times: Array,
        carrier_freqs: Array,
        carrier_phases: Array,
    ) -> Array:
        # Shape (amplitudes?, gate_times?, carrier_freqs?, carrier_phases?)
        return jnp.squeeze(
            amplitudes
            * _raised_cosine_envelope(
                t,
                gate_times=gate_times,
                carrier_freqs=carrier_freqs,
                carrier_phases=carrier_phases,
            )
        )

    return Partial(
        closure_func,
        amplitudes=amplitudes,
        gate_times=gate_times,
        carrier_freqs=carrier_freqs,
        carrier_phases=carrier_phases,
    )


def raised_cosine_gaussian_filtered(
    pixel_times: Array,
    amplitudes: PulseParamType,
    gate_times: PulseParamType,
    carrier_freqs: PulseParamType,
    carrier_phases: PulseParamType,
    gaussian_std: PulseParamType,
) -> Callable[[float], Array]:
    """Returns a Gaussian filtered raised cosine pulse function with signature
    `f(t: float) -> Array`, which nominally starts and end at 0 amplitude at `t=0.0` and
    `t=gate_time`, and reaches a maximal amplitude of `amplitudes` at `gate_time/2`.
    Of course, any finite Gaussian filter width will modify this nominal pulse shape.

    Args:
        pixel_times _(array of shape (p))_: Discretized times from which the pulse
            amplitudes are defined. `p` is the number of pixels.
        amplitudes (PulseParamType): Maximal amplitudes of the pulses.
        gate_times (PulseParamType): Total durations of the gate.
        carrier_freqs (PulseParamType): Carrier frequencies of the pulse, i.e. the
            frequencies of the fast oscillations inside the smooth cosine envelope.
            Defaults to 0.0 which corresponds to getting only the smooth envelope.
        carrier_phases (PulseParamType): Phases of the carrier at t=0.0.
        gaussian_std _(array of shape (s))_: Gaussian filter standard deviations.
            `s` is the number of different gaussian filter widths to use.

    Returns:
        callable[[float], Array]: Raised cosine pulse function. The array returned
            by this function has shape (gaussian_std?, amplitudes?, gate_times?,
            carrier_freqs?, carrier_phases?).
    """
    (
        pixel_times_expanded,
        amplitudes,
        gate_times,
        carrier_freqs,
        carrier_phases,
    ) = format_pulse_params(
        (pixel_times, amplitudes, gate_times, carrier_freqs, carrier_phases)
    )
    # Shape (Npix, time_dim?, freq_dim?, phase_dim?)
    pixel_amplitudes = jnp.squeeze(
        amplitudes
        * _raised_cosine_envelope(
            pixel_times_expanded,
            gate_times=gate_times,
            carrier_freqs=carrier_freqs,
            carrier_phases=carrier_phases,
        )
    )

    # Shape (batch_filter?, batch_amp?, batch_time?, batch_freq?, batch_phase?)
    return gaussian_filtered_func(
        pixel_times=pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        gaussian_std=gaussian_std,
    )


def gaussian_filtered_IQmixed_func(
    pixel_times: Array,
    pixel_amplitudes: Array,
    gaussian_std: PulseParamType,
    LO_frequencies: Array,
    LO_phases: Array,
) -> Callable[[float], Array]:
    mid_pixel_times, timescale, LO_frequencies, LO_phases = (
        prepare_gaussian_IQmixer_params(
            pixel_times=pixel_times,
            pixel_amplitudes=pixel_amplitudes,
            gaussian_std=gaussian_std,
            LO_frequencies=LO_frequencies,
            LO_phases=LO_phases,
        )
    )

    # Shape (batch_filter?, ...all_amplitudes_dim)
    return Partial(
        gaussian_filter_and_IQmixer_closure_func,
        mid_pixel_times=mid_pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        timescale=timescale,
        LO_frequencies=LO_frequencies,
        LO_phases=LO_phases,
    )


def raised_cosine_drag_gaussian_filtered_IQ_mixed(
    pixel_times: Array,
    amplitudes: PulseParamType,
    gate_times: PulseParamType,
    awg_frequencies: PulseParamType,
    drag_params: PulseParamType,
    LO_frequencies: PulseParamType,
    LO_phases: PulseParamType,
    gaussian_std: PulseParamType,
) -> Callable[[float], Array]:
    (
        pixel_times_expanded,
        amplitudes,
        gate_times,
        awg_frequencies,
        drag_params,
    ) = format_pulse_params(
        (pixel_times, amplitudes, gate_times, awg_frequencies, drag_params)
    )
    # Shape (Npix, time_dim?, freq_dim?, phase_dim?)
    pixel_amplitudes = jnp.squeeze(
        amplitudes
        * _raised_cosine_drag_envelope_awg(
            pixel_times_expanded,
            gate_times=gate_times,
            awg_frequencies=awg_frequencies,
            drag_params=drag_params,
        )
    )

    # Shape (batch_filter?, batch_amp?, batch_time?, batch_freq?, batch_phase?)
    return gaussian_filtered_IQmixed_func(
        pixel_times=pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        gaussian_std=gaussian_std,
        LO_frequencies=LO_frequencies,
        LO_phases=LO_phases,
    )
