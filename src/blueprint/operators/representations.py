from typing import Sequence
import math

from jax import Array
from jax import numpy as jnp

from jaxtyping import ArrayLike

from ..bases import Basis


def unitary_to_ptm(
    unitary_op: ArrayLike,
    in_bases: Basis | Sequence[Basis],
    out_bases: Basis | Sequence[Basis] | None = None,
) -> Array:
    """
    unitary_to_ptm Convert a unitary operator to its Pauli transfer matrix (PTM).

    Parameters
    ----------
    unitary_op : ArrayLike
        Unitary operator to convert. Must be a 2-D square array with shape
        ``(d, d)`` where ``d`` equals the product of the Hilbert dimensions
        of ``in_bases``.
    in_bases : Basis or sequence of Basis
        Input operator bases for the transfer matrix.
    out_bases : None or Basis or sequence of Basis, optional
        Output operator bases for the transfer matrix. If ``None``, ``in_bases``
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

    if isinstance(in_bases, Basis):
        in_bases = [in_bases]
    else:
        in_bases = list(in_bases)

    if out_bases is None:
        out_bases = in_bases
    elif isinstance(out_bases, Basis):
        out_bases = [out_bases]
    else:
        out_bases = list(out_bases)

    num_qubits = len(in_bases)
    if len(out_bases) != num_qubits:
        raise ValueError(
            f"Both input and output bases must contain the same number of basis operators ({num_qubits})."
        )

    in_dims = tuple(basis.hilbert_dim for basis in in_bases)
    out_dims = tuple(basis.hilbert_dim for basis in out_bases)

    if in_dims != out_dims:
        raise ValueError(
            f"Input and output bases must have the same Hilbert dimensions: got {in_dims} and {out_dims}."
        )

    expected_dim = math.prod(in_dims)
    if dim != expected_dim:
        raise ValueError(
            f"The input Unitary operator dimension ({dim}) does not match the"
            f" product of input basis Hilbert dimensions ({expected_dim})."
        )

    op_shape = (*in_dims, *out_dims)
    unitary_op = unitary_op.reshape(op_shape)

    pauli_inds = tuple(range(4 * num_qubits, 6 * num_qubits))

    args = []
    for ind, out_basis in enumerate(out_bases):
        args.append(out_basis.operators)
        pauli_ind = pauli_inds[ind]
        basis_ind = 2 * ind
        args.append((pauli_ind, basis_ind, basis_ind + 1))

    args.append(unitary_op)

    op_inds = tuple(range(1, 4 * num_qubits, 2))
    args.append(op_inds)

    for ind, in_basis in enumerate(in_bases):
        args.append(in_basis.operators)

        pauli_ind = pauli_inds[ind + num_qubits]
        basis_ind = 2 * (ind + num_qubits)

        args.append((pauli_ind, basis_ind + 1, basis_ind))

    conj_op = jnp.conj(unitary_op)
    args.append(conj_op)

    op_inds = tuple(range(0, 4 * num_qubits, 2))
    args.append(op_inds)

    transfer_mat = jnp.real(jnp.einsum(*args, pauli_inds, optimize="greedy"))
    return transfer_mat


def kraus_to_ptm(
    kraus_ops: ArrayLike,
    in_bases: Basis | Sequence[Basis],
    out_bases: Basis | Sequence[Basis] | None = None,
) -> Array:
    """
    kraus_to_ptm Convert a set of Kraus operators to a Pauli transfer matrix (PTM).

    Parameters
    ----------
    kraus_ops : ArrayLike
        Array of Kraus operators with shape ``(n_kraus, d, d)`` where ``d`` is
        the product of the Hilbert dimensions of ``in_bases``.
    in_bases : Basis or sequence of Basis
        Input operator bases for the transfer matrix.
    out_bases : None or Basis or sequence of Basis, optional
        Output operator bases for the transfer matrix. If ``None``, ``in_bases``
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

    num_ops, dim, other_dim = kraus_ops.shape
    if dim != other_dim:
        raise ValueError(
            f"The input Kraus operators must be square matrices, got shape {kraus_ops.shape}."
        )

    if isinstance(in_bases, Basis):
        in_bases = [in_bases]
    else:
        in_bases = list(in_bases)

    if out_bases is None:
        out_bases = in_bases
    elif isinstance(out_bases, Basis):
        out_bases = [out_bases]
    else:
        out_bases = list(out_bases)

    num_qubits = len(in_bases)
    if len(out_bases) != num_qubits:
        raise ValueError(
            f"Both input and output bases must contain the same number of basis operators ({num_qubits})."
        )

    in_dims = tuple(basis.hilbert_dim for basis in in_bases)
    out_dims = tuple(basis.hilbert_dim for basis in out_bases)

    if in_dims != out_dims:
        raise ValueError(
            f"Input and output bases must have the same Hilbert dimensions: got {in_dims} and {out_dims}."
        )

    expected_dim = math.prod(in_dims)
    if dim != expected_dim:
        raise ValueError(
            f"The input Kraus operator dimension ({dim}) does not match the"
            f" product of input basis Hilbert dimensions ({expected_dim})."
        )

    op_shape = (num_ops, *in_dims, *out_dims)
    kraus_ops = kraus_ops.reshape(op_shape)

    pauli_inds = tuple(range(4 * num_qubits, 6 * num_qubits))
    kraus_ind = 6 * num_qubits

    args = []
    for ind, out_basis in enumerate(out_bases):
        args.append(out_basis.operators)
        pauli_ind = pauli_inds[ind]
        basis_ind = 2 * ind
        args.append((pauli_ind, basis_ind, basis_ind + 1))

    args.append(kraus_ops)

    _inds = range(1, 4 * num_qubits, 2)
    op_inds = (kraus_ind, *_inds)
    args.append(op_inds)

    for ind, in_basis in enumerate(in_bases):
        args.append(in_basis.operators)

        pauli_ind = pauli_inds[ind + num_qubits]
        basis_ind = 2 * (ind + num_qubits)

        args.append((pauli_ind, basis_ind + 1, basis_ind))

    args.append(jnp.conj(kraus_ops))

    _inds = range(0, 4 * num_qubits, 2)
    op_inds = (kraus_ind, *_inds)
    args.append(op_inds)

    transfer_mat = jnp.real(jnp.einsum(*args, pauli_inds, optimize="greedy"))
    return transfer_mat
