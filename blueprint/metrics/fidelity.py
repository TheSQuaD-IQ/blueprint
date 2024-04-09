import math
from jax import Array
from jax import numpy as jnp

from ..operators.util import validate_ptm


def entanglement_fidelity(ptm_op: Array, target_ptm_op: Array) -> float:
    """
    entanglement_fidelity Calculates the entanglement fidelity between a noisy PTM operator and a target Pauli transfer operator, corresponding to a pair of noisy and target channels.

    Parameters
    ----------
    ptm_op : Array
        The noisy Pauli transfer matrix operator.

    target_ptm_op : Array
        The target Pauli transfer matrix operator.

    Returns
    -------
    float
        The entanglement fidelity between the noisy and ideal Pauli transfer matrices.

    Raises
    ------
    ValueError
        If the provided noisy Pauli transfer matrix is not 2D.
    ValueError
        If the provided noisy Pauli transfer matrix is not square.
    ValueError
        If the provided target Pauli transfer matrix is not 2D.
    ValueError
        If the provided target Pauli transfer matrix is not square.
    ValueError
        If the noisy and target Pauli transfer matrices have different dimensions.
    """
    validate_ptm(ptm_op, label="noisy")
    pauli_dim = ptm_op.shape[0]

    validate_ptm(target_ptm_op, label="target")
    target_dim = target_ptm_op.shape[0]

    if pauli_dim != target_dim:
        raise ValueError(
            f"The noisy and ideal Pauli transfer matrices must have the same dimensionality: instead the noisy and target matrix have dimensions {pauli_dim} and {target_dim}, respectively."
        )

    # NOTE: The formula for the gate fidelity Fg of a PTM operator is given in arXiv:1202.5344. Note a small typo in the formula for the Hilbert space dimension d. See also arXiv:1509.02921.
    # The gate fidelity Fg is related to the entanglement fidelity Fe by the formula Fg = (d * Fe + 1) / (d + 1), see arXiv:quant-ph/0205035v2 as well.
    conj_op = jnp.conj(target_ptm_op)
    # NOTE: that the operation above can be skipped for PTMs, but this formulat work for other superopators.
    ent_fid = jnp.trace(conj_op.T @ ptm_op) / pauli_dim
    return float(ent_fid)


def gate_fidelity(ptm_op: Array, target_ptm_op: Array) -> float:
    """
    gate_fidelity Calculates the gate fidelity between a noisy and a target Pauli transfer matrix operators.

    Parameters
    ----------
    ptm_op : Array
        The noisy Pauli transfer matrix operator.

    target_ptm_op : Array
        The target Pauli transfer matrix operator.

    Returns
    -------
    float
        The gate fidelity between the noisy and target Pauli transfer ,matrix operators.

    Raises
    ------
    ValueError
        If the provided noisy Pauli transfer matrix is not 2D.
    ValueError
        If the provided noisy Pauli transfer matrix is not square.
    ValueError
        If the provided ideal Pauli transfer matrix is not 2D.
    ValueError
        If the provided ideal Pauli transfer matrix is not square.
    ValueError
        If the noisy and ideal Pauli transfer matrices have different dimensions.
    """
    ent_fid = entanglement_fidelity(ptm_op, target_ptm_op)

    pauli_dim = ptm_op.shape[0]
    dim = int(math.sqrt(pauli_dim))
    gate_fid = (dim * ent_fid + 1) / (dim + 1)
    return gate_fid
