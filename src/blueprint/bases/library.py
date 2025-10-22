from functools import partial, lru_cache

from jax import jit, Array
from jax import numpy as jnp

from .basis import Basis


@partial(jit, static_argnames=("i", "dim"))
def get_z_basis_op(i: int, dim: int) -> Array:
    """
    get_z_basis_op Returns the non-identity diagonal operators of the generalized Gell-Mann operator basis.

    Parameters
    ----------
    i : int
        The index of the index that takes a value different from 0 or 1.
    dim : int
        The dimension of the Hilbert space of the system.

    Returns
    -------
    Array
        The diagonal operator for the given index.
    """
    diag_elements = jnp.zeros(dim, dtype=complex)
    norm_factor = jnp.sqrt(2 / (i * (i + 1)))
    diag_elements = diag_elements.at[:i].set(norm_factor)
    diag_elements = diag_elements.at[i].set(-i * norm_factor)
    operator = jnp.diag(diag_elements)
    return operator


@partial(jit, static_argnames=("i", "j", "dim"))
def get_x_basis_op(i: int, j: int, dim: int):
    """
    get_x_basis_op Returns the real symmeytic off-diagonal operators
    of the generalized Gell-Mann operator basis.

    Parameters
    ----------
    i : int
        The row index of the non-zero value of the operator.
    j : int
        The column index of the non-zero value of the operator.
    dim : int
        The dimension of the Hilbert space of the system.

    Returns
    -------
    Array
        The real off-diagonal operator for pair of non-trivial value indices.
    """
    shape = (dim, dim)
    operator = jnp.zeros(shape, dtype=complex)
    operator = operator.at[i, j].set(1)
    operator = operator.at[j, i].set(1)
    return operator


@partial(jit, static_argnames=("i", "j", "dim"))
def get_y_basis_op(i: int, j: int, dim: int) -> Array:
    """
    get_y_basis_op Returns the imaginary anti-symmetric off-diagonal operators
    of the generalized Gell-Mann operator basis.

    Parameters
    ----------
    i : int
        The row index of the non-zero value of the operator.
    j : int
        The column index of the non-zero value of the operator.
    dim : int
        The dimension of the Hilbert space of the system.

    Returns
    -------
    Array
        The imaginary off-diagonal operator for pair of non-trivial value indices.
    """
    shape = (dim, dim)
    operator = jnp.zeros(shape, dtype=complex)
    operator = operator.at[i, j].set(1j)
    operator = operator.at[j, i].set(-1j)
    return operator


@lru_cache(maxsize=8)
def get_gellmann_basis(hilbert_dim: int, normalize: bool = True) -> Basis:
    """
    get_gellmann_basis Returns the generalized Gell-Mann operator basis
    for a system with a given Hilbert space dimension hilbert_dim.

    if hilbert_dim = 2, the basis is the Pauli basis.
    if hilbert_dim = 3, the basis is the Gell-Mann basis.

    Parameters
    ----------
    hilbert_dim : int
        The dimension of the Hilbert space of the system.
    normalize : bool, optional
        Whether to normalize the basis operators, by default True

    Returns
    -------
    Basis
        The Gell-Mann operator basis.
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


@lru_cache(maxsize=8)
def get_general_basis(hilbert_dim: int, normalize: bool = True) -> Basis:
    """
    get_gellmann_basis Returns the generalized Gell-Mann operator basis
    for a system with a given Hilbert space dimension hilbert_dim.

    if hilbert_dim = 2, the basis is the Pauli basis.
    if hilbert_dim = 3, the basis is the Gell-Mann basis.

    Parameters
    ----------
    hilbert_dim : int
        The dimension of the Hilbert space of the system.
    normalize : bool, optional
        Whether to normalize the basis operators, by default True

    Returns
    -------
    Basis
        The Gell-Mann operator basis.
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
