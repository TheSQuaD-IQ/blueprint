from jax import numpy as jnp
from jaxtyping import Array, Scalar

from ..util.linalg import cosm, sinm


def get_identity_op(harmonic_cutoff: int) -> Array:
    """
    get_identity_op Return identity operator in the harmonic basis.

    Returns
    -------
    Array
        Identity matrix.
    """
    id_op = jnp.identity(harmonic_cutoff)
    return id_op


def get_raise_op(harmonic_cutoff: int) -> Array:
    """
    get_raise_op Return raising operator in the harmonic basis.

    Returns
    -------
    Array
        Creation operator matrix.
    """
    offdiag = jnp.sqrt(jnp.arange(1, harmonic_cutoff))
    raise_op = jnp.diag(offdiag, k=-1)
    return raise_op


def get_low_op(harmonic_cutoff: int) -> Array:
    """
    get_low_op Return lowering operator in the harmonic basis.

    Returns
    -------
    Array
        Lowering operator matrix.
    """
    offdiag = jnp.sqrt(jnp.arange(1, harmonic_cutoff))
    low_op = jnp.diag(offdiag, k=1)
    return low_op


def get_number_op(harmonic_cutoff: int) -> Array:
    """
    get_number_op Return number operator in the harmonic basis.

    Returns
    -------
    Array
        Number operator matrix.
    """
    diag_elems = jnp.arange(harmonic_cutoff)
    num_op = jnp.diag(diag_elems)
    return num_op


def get_charge_op(charge_zpf: Scalar, harmonic_cutoff: int) -> Array:
    """
    get_charge_op get the charge operator in the harmonic basis.

    Returns
    -------
    Array
        Charge operator in native basis.
    """
    low_op = get_low_op(harmonic_cutoff)
    raise_op = get_raise_op(harmonic_cutoff)
    charge_op = 1.0j * charge_zpf * (raise_op - low_op)
    return charge_op


def get_flux_op(flux_zpf: Scalar, harmonic_cutoff: int) -> Array:
    """
    get_flux_op get the flux operator in the harmonic basis.

    Returns
    -------
    Array
        Flux operator in native basis.
    """
    low_op = get_low_op(harmonic_cutoff)
    raise_op = get_raise_op(harmonic_cutoff)
    flux_op = flux_zpf * (raise_op + low_op)
    return flux_op


def get_cosflux_op(flux_zpf: Scalar, harmonic_cutoff: int) -> Array:
    """
    get_cosflux_op Returns the cos(flux) operator in the harmonic basis.

    Returns
    -------
    Array
        cos(flux) operator matrix.
    """
    flux_op = get_flux_op(flux_zpf, harmonic_cutoff)
    cosflux_op = cosm(flux_op)
    return cosflux_op


def get_sinflux_op(flux_zpf: Scalar, harmonic_cutoff: int) -> Array:
    """
    get_sinflux_op Returns the sin(flux) operator in the harmonic basis.

    Returns
    -------
    Array
        sin(flux) operator matrix.
    """
    flux_op = get_flux_op(flux_zpf, harmonic_cutoff)
    sinflux_op = sinm(flux_op)
    return sinflux_op
