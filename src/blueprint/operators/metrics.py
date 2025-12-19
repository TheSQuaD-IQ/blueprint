import math

from jax import jit
from jax import numpy as jnp
from jaxtyping import Scalar, Array

from ..util.linalg import dag


@jit
def _get_ent_fidelity(ptm: Array, target_ptm: Array) -> Scalar:
    """
    _get_ent_fidelity Compute the entanglement fidelity between two PTMs.

    Parameters
    ----------
    ptm : Array
        The noisy Pauli transfer matrix operator (shape ``(N, N)``).
    target_ptm : Array
        The target Pauli transfer matrix operator (shape ``(N, N)``).

    Returns
    -------
    Scalar
        The entanglement fidelity (real scalar, possibly a 0-d JAX Array).

    Notes
    -----
    The formula for the gate fidelity Fg of a PTM operator is given in
    arXiv:1202.5344 (see also arXiv:1509.02921). The gate fidelity Fg is related
    to the entanglement fidelity Fe by Fg = (d * Fe + 1) / (d + 1).
    """
    pauli_dim = ptm.shape[0]
    ent_fid = jnp.real(jnp.trace(dag(target_ptm) @ ptm)) / pauli_dim
    return ent_fid


def get_ent_fidelity(ptm: Array, target_ptm: Array) -> Scalar:
    """
    get_ent_fidelity Return the entanglement fidelity between two PTMs.

    Parameters
    ----------
    ptm : Array
        The noisy Pauli transfer matrix operator (shape ``(N, N)``).
    target_ptm : Array
        The target Pauli transfer matrix operator (shape ``(N, N)``).

    Returns
    -------
    Scalar
        The entanglement fidelity between the noisy and ideal Pauli transfer matrices.
    """
    pauli_dim, other_dim = ptm.shape
    if pauli_dim != other_dim:
        raise ValueError("The input PTM operator must be a square matrix.")

    # NOTE: The formula for the gate fidelity Fg of a PTM operator is given in arXiv:1202.5344. Note a small typo in the formula for the Hilbert space dimension d. See also arXiv:1509.02921.
    # The gate fidelity Fg is related to the entanglement fidelity Fe by the formula Fg = (d * Fe + 1) / (d + 1), see arXiv:quant-ph/0205035v2 as well.
    ent_fid = _get_ent_fidelity(ptm, target_ptm)
    return ent_fid


def get_gate_fidelity(
    ptm: Array,
    target_ptm: Array,
    leakage_rate: float | Scalar = 0.0,
) -> Scalar:
    """
    get_gate_fidelity Return the average gate fidelity between a noisy and target PTM.

    Parameters
    ----------
    ptm : Array
        The noisy Pauli transfer matrix operator (shape ``(N, N)``).
    target_ptm : Array
        The target Pauli transfer matrix operator (shape ``(N, N)``).
    leakage_rate : float or Scalar, optional
        The leakage rate of the noisy channel. Defaults to ``0.0``.

    Returns
    -------
    Scalar
        The gate fidelity between the noisy and target Pauli transfer matrix operators.
    """
    pauli_dim, other_dim = ptm.shape
    if pauli_dim != other_dim:
        raise ValueError("The input PTM operator must be a square matrix.")

    if ptm.shape != target_ptm.shape:
        raise ValueError("The input PTM operators must have the same shape.")

    ent_fid = _get_ent_fidelity(ptm, target_ptm)

    dim = int(math.sqrt(pauli_dim))
    gate_fidelity = (dim * ent_fid + 1 - leakage_rate) / (dim + 1)
    return gate_fidelity
