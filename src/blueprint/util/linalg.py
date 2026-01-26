from operator import mul
from functools import reduce
from typing import Sequence

from jax import Array
from jax import numpy as jnp
from jax.tree_util import tree_reduce
from jax.scipy.linalg import expm


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
    cosm_op = 0.5 * (expm(1j * op) + expm(-1j * op))
    return cosm_op


def sinm(op: Array) -> Array:
    """sinm Returns the matrix sine of the given operator.

    Parameters
    ----------
    op : Array
        The operator to take the matrix sine of.

    Returns
    -------
    Array
        The matrix sine of the operator.
    """
    sinm_op = -0.5j * (expm(1j * op) - expm(-1j * op))
    return sinm_op


def dag(operator: Array) -> Array:
    """
    dag Returns the conjugate transpose of the input operator.

    Parameters
    ----------
    operator : Array
        The operator whose conjugate transpose is to be computed.

    Returns
    -------
    Array
        The conjugate-transposed input operator.
    """
    return jnp.conj(jnp.matrix_transpose(operator))


def tensor_product(*ops: Array) -> Array:
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
    return tree_reduce(jnp.kron, ops)


def matrix_product(ops: Sequence[Array]) -> Array:
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
    return tree_reduce(jnp.matmul, ops)


def transform_op(op: Array, transform_mat: Array) -> Array:
    """
    transform_op Performs a unitary transformation on the input operator.

    Parameters
    ----------
    op : Array
        The operator to be transformed.
    transform_mat : Array
        The transformation matrix.

    Returns
    -------
    Array
        The transformed operator.
    """
    return dag(transform_mat) @ op @ transform_mat


def embed_op(
    op: Array,
    ind: int,
    dims: Sequence[int],
) -> Array:
    """
    embed_operator Embeds an operator into a larger Hilbert space.

    Parameters
    ----------
    op : Array
        The operator to embed.
    ind : int
        The index of the qubit in the larger Hilbert space.
    dims : Sequence[int]
        The Hilbert-space dimensions for each subsystem (sequence of ints).

    Returns
    -------
    Array
        The embedded operator.
    """
    next_ind = ind + 1
    left_dim = reduce(mul, dims[:ind], 1)
    right_dim = reduce(mul, dims[next_ind:], 1)

    left_identity = jnp.identity(left_dim)
    right_identity = jnp.identity(right_dim)

    embedded_op = jnp.kron(op, right_identity)
    embedded_op = jnp.kron(left_identity, embedded_op)
    return embedded_op


def tidyup(operator: Array, tol: float) -> Array:
    """tidyup Tidies up the input matrix by truncating small values to zero.
    The real and imaginary parts are treated separately. For example, the number
    `1e-18 + 2j` will be truncated with a tolerance of `1e-15` to just `2j`.

    Note:
        Code adapted from
        https://github.com/qutip/qutip/blob/master/qutip/core/data/tidyup.pyx.

    Parameters
    ----------
    operator : Array
        The operator to tidy up.
    tol: float
        Absolute tolerance of the smallest real or imag values to keep in the Array.

    Returns
    -------
    Array
        The tidied-up array containing only real and imaginary values larger than `tol`.
    """
    real_op = jnp.real(operator)
    real_op = jnp.where(jnp.abs(real_op) < tol, 0.0, real_op)

    imag_op = jnp.imag(operator)
    imag_op = jnp.where(jnp.abs(imag_op) < tol, 0.0, imag_op)
    return real_op + 1.0j * imag_op
