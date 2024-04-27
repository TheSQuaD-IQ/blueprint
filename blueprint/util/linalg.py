"""Module for linear algebra utilities."""

from typing import Iterable, List

from collections import deque
from functools import reduce


from jax import numpy as jnp
from jax import scipy as jsp
from jax import Array


def cosm(op: Array) -> Array:
    """
    cosm Returns the matrix cosine of the given operator.

    Parameters
    ----------
    op : Array
        The operator to take the matrix cosine of.

    Returns
    -------
    Array
        The matrix cosine of the operator.
    """
    exponent = 1.0j * op
    cosm_op = 0.5 * (jsp.linalg.expm(exponent) + jsp.linalg.expm(-exponent))
    return cosm_op


def transform_op(op: Array, trans_op: Array) -> Array:
    """
    transform_op Transforms an operator to a new basis.

    Parameters
    ----------
    op : Array
        The operator to transform.
    transformation : Array
        The transformation matrix.

    Returns
    -------
    Array
        The transformed operator.
    """
    transformed_op = jnp.conj(trans_op).T @ op @ trans_op
    return transformed_op


def tensor_product(ops: Iterable[Array]) -> Array:
    """
    tensor_product Returns the tensor product of a list of operators.

    Parameters
    ----------
    ops : Iterable[Array]
        The list of operators to tensor product.

    Returns
    -------
    Array
        The tensor product of the operators.
    """
    return reduce(jnp.kron, ops)


def matrix_product(ops: Iterable[Array]) -> Array:
    """
    matrix_product Returns the matrix product of a list of operators.

    Parameters
    ----------
    ops : Iterable[Array]
        The list of operators to matrix product.

    Returns
    -------
    Array
        The matrix product of the operators.
    """
    return reduce(jnp.matmul, ops)


def embed_op(
    op: Array,
    ind: int,
    dims: Iterable[int],
) -> Array:
    """
    embed_operator Embeds an operator into a larger Hilbert space.

    Parameters
    ----------
    op : Array
        The operator to embed.
    ind : int
        The index of the qubit in the larger Hilbert space.
    dims : int
        The dimension of the qubit in the larger Hilbert space.

    Returns
    -------
    Array
        The embedded operator.
    """
    expanded_ops = (op if i == ind else jnp.identity(dim) for i, dim in enumerate(dims))
    return tensor_product(expanded_ops)


def embed_ops(
    ops: List[Array],
    inds: List[int],
    dims: Iterable[int],
) -> Array:
    """
    embed_ops Embeds operators into a larger Hilbert space.

    Parameters
    ----------
    ops : Iterable[Array]
        The operators to embed.
    inds : Iterable[int]
        The index of the qubit in the larger Hilbert space.
    dims : Iterable[int]
        The dimensions of the qubits in the larger Hilbert space.

    Returns
    -------
    Array
        The embedded operators.
    """
    expanded_ops = deque()

    for ind, dim in enumerate(dims):
        if ind in inds:
            idx = inds.index(ind)
            expanded_ops.append(ops[idx])
        else:
            expanded_ops.append(jnp.identity(dim))
    return tensor_product(expanded_ops)


def dag(op: Array) -> Array:
    """
    dag Returns the conjugate transpose of the operator.

    Parameters
    ----------
    op : Array
        The operator to conjugate transpose.

    Returns
    -------
    Array
        The conjugate transpose of the operator.
    """
    return jnp.transpose(jnp.conj(op))


def is_hermitian(
    op: Array, *, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    """
    is_hermitian Checks if the provided operator is Hermitian.

    Parameters
    ----------
    op : Array
        The operator to check for Hermiticity.

    Returns
    -------
    bool
        Whether the provided operator is Hermitian.
    """
    conj_op = jnp.transpose(jnp.conj(op))
    result = jnp.allclose(op, conj_op, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return bool(result)


def is_diagonal(
    op: Array, *, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    """
    is_diagonal Checks if the provided operator is diagonal.

    Parameters
    ----------
    op : Array
        The operator to check for diagonalization.

    Returns
    -------
    bool
        Whether the provided operator is diagonal.
    """
    diag_elements = jnp.diag(op)
    diag_op = jnp.diag(diag_elements)
    result = jnp.allclose(op, diag_op, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return bool(result)
