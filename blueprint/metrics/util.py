from typing import Any

from jax import Array
from jax import numpy as jnp


def validate_ptm(ptm_op: Array, label: str | None = None) -> None:
    """
    validate_ptm Validates the provided Pauli transfer matrix.

    Parameters
    ----------
    ptm_op : Array
        The Pauli transfer matrix to validate.
    label : str | None, optional
        The label for the pauli transfer matrix used for raising exceptions, by default None

    Raises
    ------
    ValueError
        If the provided Pauli transfer matrix is not 2D.
    ValueError
        If the provided Pauli transfer matrix is not square.
    """
    op_label = f"{label} superoperator" if label else "superoperator"

    num_dim = len(ptm_op.shape)
    if num_dim != 2:
        raise ValueError(
            f"Only 2D Pauli transfer matrices are accepted: the provided {op_label} is {num_dim}-dimensional."
        )

    pauli_dim, other_dim = ptm_op.shape
    if pauli_dim != other_dim:
        raise ValueError(
            f"Only square Pauli transfer matrices are accepted: the provided {op_label} is ({pauli_dim}, {other_dim}) dimensional."
        )

    if not all(jnp.isreal(ptm_op)):
        raise ValueError(
            f"The Pauli transfer matrices is expected to be real: the provided {op_label} is complex-valued."
        )


def is_trace_preserving(
    ptm_op: Array,
    label: str | None = None,
    **keywords: Any,
) -> bool:
    """
    is_trace_preserving Checks if the provided Pauli transfer matrix is trace preserving

    Parameters
    ----------
    ptm_op : Array
        The Pauli transfer matrix to check for trace preservation.
    label : str | None, optional
        The optional label of the PTM that is ued for exception raising in case an invalid PTM is provided, by default None

    Returns
    -------
    bool
        Whether the provided PTM is trace preserving
    """
    validate_ptm(ptm_op, label=label)
    pauli_dim = ptm_op.shape[0]
    ptm_row = ptm_op[0]

    expected_row = jnp.zeros(pauli_dim, dtype=ptm_op.dtype)
    expected_row[0] = 1

    result = jnp.allclose(ptm_row, expected_row, **keywords)
    return bool(result)


def is_unital(
    ptm_op: Array,
    label: str | None = None,
    **keywords: Any,
) -> bool:
    """
    is_unital Cheks if the provided Pauli transfer matrix is unital.

    Parameters
    ----------
    ptm_op : Array
        The Pauli transfer matrix to check for unitality.
    label : str | None, optional
        The optional label of the PTM that is ued for exception raising in case an invalid PTM is provided, by default None

    Returns
    -------
    bool
        Whether the provided PTM is unital.
    """
    validate_ptm(ptm_op, label=label)
    pauli_dim = ptm_op.shape[0]
    ptm_col = ptm_op[:, 0]

    expected_col = jnp.zeros(pauli_dim, dtype=ptm_op.dtype)
    expected_col[0] = 1

    result = jnp.allclose(ptm_col, expected_col, **keywords)
    return bool(result)
