from typing import Iterable, List

from collections import deque
from functools import reduce


from jax import numpy as jnp
from jax import Array


def transform_op(op: Array, trans_mat: Array) -> Array:
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
    trans_op = jnp.conj(trans_mat).T @ op @ trans_mat
    return trans_op

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

def embed_op(
    op: Array,
    ind: int,
    qubit_dims: Iterable[int],
) -> Array:
    """
    embed_operator Embeds an operator into a larger Hilbert space.

    Parameters
    ----------
    op : Array
        The operator to embed.
    ind : int
        The index of the qubit in the larger Hilbert space.
    qubit_dim : int
        The dimension of the qubit in the larger Hilbert space.

    Returns
    -------
    Array
        The embedded operator.
    """
    expanded_ops = (
        op if i == ind else jnp.identity(dim) for i, dim in enumerate(qubit_dims)
    )
    return tensor_product(expanded_ops)


def embed_ops(
    ops: List[Array],
    inds: List[int],
    qubit_dims: Iterable[int],
) -> Array:
    """
    embed_ops Embeds operators into a larger Hilbert space.

    Parameters
    ----------
    ops : Iterable[Array]
        The operators to embed.
    inds : Iterable[int]
        The index of the qubit in the larger Hilbert space.
    qubit_dims : Iterable[int]
        The dimensions of the qubits in the larger Hilbert space.

    Returns
    -------
    Array
        The embedded operators.
    """
    expanded_ops = deque()

    for ind, dim in enumerate(qubit_dims):
        if ind in inds:
            idx = inds.index(ind)
            expanded_ops.append(ops[idx])
        else:
            expanded_ops.append(jnp.identity(dim))
    return tensor_product(expanded_ops)
