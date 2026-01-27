from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Scalar, Array


def get_identity_op(charge_cutoff: int) -> Array:
    """
    get_identity_op Return identity operator in the native charge basis.

    Returns
    -------
    Array
        Identity matrix for the native charge basis.
    """
    charge_dim = 2 * charge_cutoff + 1
    id_op = jnp.identity(charge_dim)
    return id_op


def get_charge_op(charge_offset: Scalar, charge_cutoff: int) -> Array:
    """
    get_charge_op Construct the charge operator including offset in the charge basis.

    Returns
    -------
    Array
        Charge operator in the native charge basis.
    """
    charge_vals = jnp.arange(-charge_cutoff, charge_cutoff + 1)
    charge_op = jnp.diag(charge_vals)

    id_op = get_identity_op(charge_cutoff)
    offset_op = charge_offset * id_op

    offset_charge_op = charge_op - offset_op
    return offset_charge_op


def get_cosflux_op(charge_cutoff: int) -> Array:
    """
    get_cosflux_op Construct the native cos(flux) operator in the charge basis.

    Returns
    -------
    Array
        cos(flux) operator in native basis.
    """
    offdiag_elems = jnp.ones(2 * charge_cutoff)
    superdiag_mat = jnp.diag(0.5 * offdiag_elems, 1)
    subdiag_mat = jnp.transpose(superdiag_mat)
    op = superdiag_mat + subdiag_mat
    return op


def get_sinflux_op(charge_cutoff: int) -> Array:
    """
    get_sinflux_op Construct the sin(flux) operator in the charge basis.

    Returns
    -------
    Array
        sin(flux) operator in native basis.
    """
    offdiag_elems = jnp.ones(2 * charge_cutoff)
    superdiag_mat = jnp.diag(0.5j * offdiag_elems, 1)
    subdiag_mat = jnp.transpose(superdiag_mat)
    op = superdiag_mat - subdiag_mat
    return op
