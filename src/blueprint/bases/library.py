from functools import partial

from jax import jit, Array
from jax import numpy as jnp

from .basis import Basis


@partial(jit, static_argnames=("i", "dim"))
def get_z_basis_op(i: int, dim: int) -> Array:
    """
    get_z_basis_op Return a non-identity diagonal generalized Gell-Mann operator.

    Parameters
    ----------
    i : int
        Index selecting which diagonal operator to construct.
    dim : int
        Hilbert-space dimension.

    Returns
    -------
    Array
        Diagonal operator (shape ``(dim, dim)``).
    """
    diag_elements = jnp.zeros(dim, dtype=complex)
    norm_factor = jnp.sqrt(2 / (i * (i + 1)))
    diag_elements = diag_elements.at[:i].set(norm_factor)
    diag_elements = diag_elements.at[i].set(-i * norm_factor)
    operator = jnp.diag(diag_elements)
    return operator


@partial(jit, static_argnames=("i", "j", "dim"))
def get_x_basis_op(i: int, j: int, dim: int) -> Array:
    """
    get_x_basis_op Return a symmetric real off-diagonal generalized Gell-Mann operator.

    Parameters
    ----------
    i, j : int
        Row and column indices for the non-zero entries (i < j).
    dim : int
        Hilbert-space dimension.

    Returns
    -------
    Array
        Off-diagonal operator (shape ``(dim, dim)``).
    """
    shape = (dim, dim)
    operator = jnp.zeros(shape, dtype=complex)
    operator = operator.at[i, j].set(1)
    operator = operator.at[j, i].set(1)
    return operator


@partial(jit, static_argnames=("i", "j", "dim"))
def get_y_basis_op(i: int, j: int, dim: int) -> Array:
    """
    get_y_basis_op Return an anti-symmetric imaginary off-diagonal generalized Gell-Mann operator.

    Parameters
    ----------
    i, j : int
        Row and column indices for the non-zero entries (i < j).
    dim : int
        Hilbert-space dimension.

    Returns
    -------
    Array
        Off-diagonal operator (shape ``(dim, dim)``).
    """
    shape = (dim, dim)
    operator = jnp.zeros(shape, dtype=complex)
    operator = operator.at[i, j].set(1j)
    operator = operator.at[j, i].set(-1j)
    return operator


def get_gellmann_basis(hilbert_dim: int, normalize: bool = True) -> Basis:
    """
    get_gellmann_basis Construct the generalized Gell-Mann basis for given Hilbert dimension.

    Parameters
    ----------
    hilbert_dim : int
        Hilbert-space dimension.
    normalize : bool, optional
        If True, normalize basis operators to unit norm.

    Returns
    -------
    Basis
        Operator basis instance.
    """
    operator_list = []
    labels = []

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

            label = f"λ{row_ind}{col_ind}"
            labels.append(label)

    operators = jnp.stack(operator_list)

    if normalize:
        inner_prods = jnp.einsum("aij, aji -> a", operators, operators)
        op_norms = jnp.sqrt(inner_prods)
        norm_operators = operators / op_norms[:, None, None]
        basis = Basis(norm_operators, labels)
        return basis

    basis = Basis(operators, labels)
    return basis


def get_general_basis(hilbert_dim: int, normalize: bool = True) -> Basis:
    """
    get_general_basis Construct a general operator basis for given Hilbert dimension.

    Parameters
    ----------
    hilbert_dim : int
        Hilbert-space dimension.
    normalize : bool, optional
        If True, normalize basis operators to unit norm.

    Returns
    -------
    Basis
        Operator basis instance.
    """
    shape = (hilbert_dim, hilbert_dim, hilbert_dim)
    diag_ops = jnp.zeros(shape, dtype=complex)
    diag_ops = jnp.fill_diagonal(diag_ops, 1.0, inplace=False)

    inds = range(hilbert_dim)
    labels = list(map(str, inds))

    offdiag_op_list = []

    for row_ind in inds:
        for col_ind in range(row_ind):
            x_basis_op = get_x_basis_op(row_ind, col_ind, hilbert_dim)
            offdiag_op_list.append(x_basis_op)

            x_label = f"X{row_ind}{col_ind}"
            labels.append(x_label)

            y_basis_op = get_y_basis_op(row_ind, col_ind, hilbert_dim)
            offdiag_op_list.append(y_basis_op)

            y_label = f"Y{row_ind}{col_ind}"
            labels.append(y_label)

    offdiag_ops = jnp.stack(offdiag_op_list)
    operators = jnp.concatenate((diag_ops, offdiag_ops))

    if normalize:
        inner_prods = jnp.einsum("aij, aji -> a", operators, operators)
        op_norms = jnp.sqrt(inner_prods)
        norm_operators = operators / op_norms[:, None, None]
        basis = Basis(norm_operators, labels)
        return basis

    basis = Basis(operators, labels)
    return basis


def get_pauli_basis(normalize: bool = True) -> Basis:
    """
    get_pauli_basis Return the single-qubit Pauli operator basis.

    Parameters
    ----------
    normalize : bool, optional
        If True, normalize operators to unit norm.

    Returns
    -------
    Basis
        Pauli operator basis instance.
    """
    basis = get_gellmann_basis(hilbert_dim=2, normalize=normalize)
    return basis
