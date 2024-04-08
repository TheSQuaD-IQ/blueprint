from jax import Array
from jax import numpy as jnp


def kraus_to_ptm(
    kraus_ops: Array, input_basis_ops: Array, output_basis_ops: Array
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
    dim = kraus_ops.shape[1]
    ptm = jnp.einsum(
        "iab, kbc, jcd, kad -> ij",
        input_basis_ops,
        kraus_ops,
        output_basis_ops,
        jnp.conjugate(kraus_ops),
    )
    return ptm.real / dim
