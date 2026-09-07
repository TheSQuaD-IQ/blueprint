from __future__ import annotations

from functools import partial

from jax import jit
from jax import numpy as jnp
from jaxtyping import Scalar, Array


@partial(jit, static_argnames=("charge_cutoff",))
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


@partial(jit, static_argnames=("charge_cutoff", "add_charge_offset"))
def get_charge_op(
    charge_cutoff: int, charge_offset: Scalar = 0.0, add_charge_offset: bool = False
) -> Array:
    """
    get_charge_op Construct the charge operator including offset in the charge basis.

    Parameters
    ----------
    charge_cutoff : int
        Charge cutoff of the native basis. Static under ``jit``.
    charge_offset : Scalar, optional
        Offset charge ``n_g``, traced under ``jit``. Ignored unless
        ``add_charge_offset`` is True. Defaults to 0.0.
    add_charge_offset : bool, optional
        Whether to subtract ``n_g`` from the charge operator. Must be a concrete
        Python bool (it is a static argument), never a traced array. Defaults to False.

    Returns
    -------
    Array
        Charge operator in the native charge basis.
    """
    charge_vals = jnp.arange(-charge_cutoff, charge_cutoff + 1)
    charge_op = jnp.diag(charge_vals)

    if add_charge_offset:
        id_op = get_identity_op(charge_cutoff)
        offset_op = charge_offset * id_op
        charge_op = charge_op - offset_op
    return charge_op


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
