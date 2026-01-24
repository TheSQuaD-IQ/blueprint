from jax import numpy as jnp
from jaxtyping import Array

from ..bases import Basis


def unitary_to_ptm(
    unitary_op: Array,
    in_basis: Basis,
    out_basis: Basis | None = None,
) -> Array:
    """
    unitary_to_ptm Convert a unitary operator to its Pauli transfer matrix (PTM).

    Parameters
    ----------
    unitary_op : Array
        Unitary operator to convert. Must be a 2-D square array with shape
        ``(d, d)`` where ``d`` equals the product of the Hilbert dimensions
        of ``in_basis``.
    in_basis : Basis
        Input operator basis for the transfer matrix.
    out_basis : None or Basis, optional
        Output operator basis for the transfer matrix. If ``None``, ``in_basis``
        is used.

    Returns
    -------
    Array
        The Pauli transfer matrix representation of the unitary operator.

    Raises
    ------
    ValueError
        If input and output bases differ in number of basis operators or in
        Hilbert dimensions, or if the matrix size does not match the expected
        Hilbert-space product.
    """
    unitary_op = jnp.asarray(unitary_op)

    if unitary_op.ndim != 2:
        raise ValueError(
            f"The input Unitary operator must be a 2D array, got shape {unitary_op.shape}."
        )

    dim, other_dim = unitary_op.shape
    if dim != other_dim:
        raise ValueError(
            f"The input Unitary operator must be a square matrix, got shape {unitary_op.shape}."
        )

    if not isinstance(in_basis, Basis):
        raise ValueError("in_basis must be an instance of Basis.")

    if out_basis is None:
        out_basis = in_basis
    elif not isinstance(out_basis, Basis):
        raise ValueError("out_basis must be an instance of Basis.")

    if in_basis.hilbert_dim != out_basis.hilbert_dim:
        raise ValueError(
            f"Input and output bases must have the same Hilbert dimensions: got {in_basis.hilbert_dim} and {out_basis.hilbert_dim}."
        )

    if dim != in_basis.hilbert_dim:
        raise ValueError(
            f"The input Kraus operator dimension ({dim}) does not match the"
            f" product of input basis Hilbert dimensions ({in_basis.hilbert_dim})."
        )

    transfer_mat = jnp.einsum(
        "aij, jl, blk, ik -> ab",
        out_basis.operators,
        unitary_op,
        in_basis.operators,
        jnp.conj(unitary_op),
        optimize="greedy",
    )

    return jnp.real(transfer_mat)


def kraus_to_ptm(
    kraus_ops: Array,
    in_basis: Basis,
    out_basis: Basis | None = None,
) -> Array:
    """
    kraus_to_ptm Convert a set of Kraus operators to a Pauli transfer matrix (PTM).

    Parameters
    ----------
    kraus_ops : Array
        Array of Kraus operators with shape ``(n_kraus, d, d)`` where ``d`` is
        the product of the Hilbert dimensions of ``in_basis``.
    in_basis : Basis
        Input operator basis for the transfer matrix.
    out_basis : None or Basis, optional
        Output operator basis for the transfer matrix. If ``None``, ``in_basis``
        is used.

    Returns
    -------
    Array
        The Pauli transfer matrix representation of the quantum channel
        described by the Kraus operators.

    Raises
    ------
    ValueError
        If the input and output bases differ in number of basis operators or
        Hilbert dimensions, or if the Kraus operator matrix size does not match
        the expected Hilbert-space product.
    """
    kraus_ops = jnp.asarray(kraus_ops)

    if kraus_ops.ndim != 3:
        raise ValueError(
            f"The input Kraus operators must be a 3D array, got shape {kraus_ops.shape}."
        )

    _, dim, other_dim = kraus_ops.shape
    if dim != other_dim:
        raise ValueError(
            f"The input Kraus operators must be square matrices, got shape {kraus_ops.shape}."
        )

    if not isinstance(in_basis, Basis):
        raise ValueError("in_basis must be an instance of Basis.")

    if out_basis is None:
        out_basis = in_basis
    elif not isinstance(out_basis, Basis):
        raise ValueError("out_basis must be an instance of Basis.")

    if in_basis.hilbert_dim != out_basis.hilbert_dim:
        raise ValueError(
            f"Input and output bases must have the same Hilbert dimensions: got {in_basis.hilbert_dim} and {out_basis.hilbert_dim}."
        )

    if dim != in_basis.hilbert_dim:
        raise ValueError(
            f"The input Kraus operator dimension ({dim}) does not match the"
            f" product of input basis Hilbert dimensions ({in_basis.hilbert_dim})."
        )

    transfer_mat = jnp.einsum(
        "aij, cjl, blk, cik -> ab",
        out_basis.operators,
        kraus_ops,
        in_basis.operators,
        jnp.conj(kraus_ops),
        optimize="greedy",
    )

    return jnp.real(transfer_mat)
