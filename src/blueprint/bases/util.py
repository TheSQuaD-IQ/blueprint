from jax import numpy as jnp
from jaxtyping import Array

from .basis import Basis
from .library import get_x_basis_op, get_y_basis_op, get_z_basis_op


def transform_basis(basis: Basis, transform_op: Array) -> Basis:
    _, dim = transform_op.shape
    if dim != basis.hilbert_dim:
        raise ValueError(
            f"transform_op has incompatible shape for a basis with a Hilbert space dimension {basis.hilbert_dim}."
        )

    transformed_ops = jnp.einsum(
        "ni, aij, mj -> anm", transform_op, basis.operators, jnp.conj(transform_op)
    )
    basis = Basis(transformed_ops, basis.labels)
    return basis


def get_pauli_ops(normalize: bool = True) -> Array:
    """
    get_pauli_ops Returns the Pauli operators for a single qubit system.

    Parameters
    ----------
    normalize : bool, optional
        Whether to normalize the basis operators, by default True

    Returns
    -------
    Array
        The Pauli operators.
    """
    hilbert_dim = 2
    operator_list = []

    for row_ind in range(hilbert_dim):
        for col_ind in range(hilbert_dim):
            if row_ind == col_ind:
                if row_ind == 0:
                    operator = jnp.identity(hilbert_dim)
                else:
                    operator = get_z_basis_op(row_ind, hilbert_dim)
            elif row_ind < col_ind:
                operator = get_x_basis_op(row_ind, col_ind, hilbert_dim)
            else:
                operator = get_y_basis_op(row_ind, col_ind, hilbert_dim)

            operator_list.append(operator)

    operators = jnp.stack(operator_list)

    if normalize:
        inner_prods = jnp.einsum("aij, aji -> a", operators, operators)
        op_norms = jnp.sqrt(inner_prods)
        norm_operators = operators / op_norms[:, None, None]
        return norm_operators

    return operators
