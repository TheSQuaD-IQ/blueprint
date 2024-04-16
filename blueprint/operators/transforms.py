from jax import Array
from jax import numpy as jnp

from .basis import OperatorBasis


def unitary_to_ptm(
    unitary: Array, input_basis: OperatorBasis, output_basis: OperatorBasis
) -> Array:
    """
    unitary_to_ptm Converts a unitary operator to a Pauli transfer matrix (PTM) superoperator.

    Parameters
    ----------
    unitary : Array
        The unitary operator.
    input_basis : OperatorBasis
        The input operator basis.
    output_basis : OperatorBasis
        The output operator basis.

    Returns
    -------
    Array
        The PTM superoperator.

    Raises
    ------
    ValueError
        If unitary is not a jax.numpy.ndarray.
    ValueError
        If unitary is not a 2D array.
    ValueError
        If unitary is not a square matrix.
    ValueError
        If input_basis is not an OperatorBasis object.
    ValueError
        If input_basis does not have the same dimension as unitary.
    ValueError
        If output_basis is not an OperatorBasis object.
    ValueError
        If output_basis does not have the same dimension as unitary.
    """
    if not isinstance(unitary, Array):
        raise ValueError("unitary must be a jax.numpy.ndarray")
    num_dims = len(unitary.shape)
    if num_dims != 2:
        raise ValueError("unitary must be a 2D array")
    dim, other_dim = unitary.shape
    if dim != other_dim:
        raise ValueError("unitary must be a square matrix")

    if not isinstance(input_basis, OperatorBasis):
        raise ValueError("input_basis must be an OperatorBasis object")
    if input_basis.dim != dim:
        raise ValueError("input_basis must have the same dimension as unitary")

    if not isinstance(output_basis, OperatorBasis):
        raise ValueError("output_basis must be an OperatorBasis object")
    if output_basis.dim != dim:
        raise ValueError("output_basis must have the same dimension as unitary")

    ptm = jnp.einsum(
        "iab, bc, jcd, ad -> ij",
        input_basis.operators,
        unitary,
        output_basis.operators,
        jnp.conjugate(unitary),
    )
    return ptm.real / dim


def kraus_to_ptm(
    kraus_ops: Array, input_basis: OperatorBasis, output_basis: OperatorBasis
) -> Array:
    """
    kraus_to_ptm Converts a set of Kraus operators to a Pauli transfer matrix (PTM) superoperator.
    The formula that the Einstein summation below is implementing is given
    R_{i,j} = Tr(P_{i}S(P_{j})), where R is the PTM operator, P_{i} and P_{j} are the input and output basis operators, respectively, and S is the a linear map defined by the Kraus operators.
    In particular, S(o) = \sum_k K_{k} o K_{k}^{\dagger}, where K_{k} are the Kraus operators.
    Combining these expressions, we get R_{i,j} = Tr(P_{i} \sum_k K_{k} P_{j} K_{k}^{\dagger}).
    See arXiv:1509.02921 for more information.

    Parameters
    ----------
    kraus_ops : Array
        The Kraus operators.
    input_basis_ops : Array
        The input basis operators.
    output_basis_ops : Array
        The output basis operators.

    Returns
    -------
    Array
        The PTM superoperator.
    """
    if not isinstance(kraus_ops, Array):
        raise ValueError("unitary must be a jax.numpy.ndarray")
    num_dims = len(kraus_ops.shape)
    if num_dims == 2:
        ops = jnp.expand_dims(kraus_ops, axis=0)
    elif num_dims == 3:
        ops = kraus_ops
    else:
        raise ValueError(
            f"kraus_ops must be a 2D or 3D array, instead got a {num_dims}D array."
        )
    num_kraus, dim, other_dim = ops.shape
    if dim != other_dim:
        raise ValueError(
            f"Each kraus operaator must be a square matrix, instead got a shape ({num_kraus},{dim},{other_dim})"
        )
    if not isinstance(input_basis, OperatorBasis):
        raise ValueError("input_basis must be an OperatorBasis object")
    if input_basis.dim != dim:
        raise ValueError("input_basis must have the same dimension as unitary")

    if not isinstance(output_basis, OperatorBasis):
        raise ValueError("output_basis must be an OperatorBasis object")
    if output_basis.dim != dim:
        raise ValueError("output_basis must have the same dimension as unitary")

    dim = kraus_ops.shape[1]
    ptm = jnp.einsum(
        "iab, kbc, jcd, kad -> ij",
        input_basis.operators,
        kraus_ops,
        output_basis.operators,
        jnp.conjugate(kraus_ops),
    )
    return ptm.real / dim
