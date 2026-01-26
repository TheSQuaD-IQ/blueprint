from jax import numpy as jnp
from jaxtyping import Scalar, Array

from .representations import unitary_to_ptm
from ..bases import Basis

type Float = float | Scalar


def get_hadamard(basis: Basis) -> Array:
    """
    get_hadamard Returns the Hadamard operation Pauli transfer matrix.

    Returns
    -------
    Array
        The Hadamard Pauli transfer matrix.
    """
    inv_sqrt2 = 1 / jnp.sqrt(2)
    unitary_op = jnp.array([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]])
    ptm = unitary_to_ptm(unitary_op, basis, basis)
    return ptm


def get_x_rotation(angle: Float, basis: Basis) -> Array:
    """
    get_x_rotation Returns a rotation operation by a given angle
    around the x-axis.

    Parameters
    ----------
    angle : Float
        The angle of rotation around the x-axis.

    Returns
    -------
    Array
        The x-rotation Pauli transfer matrix.
    """
    isin = 1j * jnp.sin(0.5 * angle)
    cos = jnp.cos(0.5 * angle)
    unitary_op = jnp.array([[cos, -isin], [-isin, cos]])
    ptm = unitary_to_ptm(unitary_op, basis, basis)
    return ptm


def get_y_rotation(angle: Float, basis: Basis) -> Array:
    """
    get_y_rotation Returns a rotation operation by a given angle
    around the y-axis.

    Parameters
    ----------
    angle : Float
        The angle of rotation around the y-axis.

    Returns
    -------
    Array
        The y-rotation Pauli transfer matrix.
    """
    sin = jnp.sin(0.5 * angle)
    cos = jnp.cos(0.5 * angle)
    unitary_op = jnp.array([[cos, -sin], [sin, cos]])
    ptm = unitary_to_ptm(unitary_op, basis, basis)
    return ptm


def get_phase_shift(phase: Float, basis: Basis) -> Array:
    """
    get_phase_shift Returns a phase shift operation by a given angle.

    Parameters
    ----------
    phase : Float
        The phase shift angle.

    Returns
    -------
    Array
        The phase shift Pauli transfer matrix.
    """
    exp = jnp.exp(1j * phase)
    unitary_op = jnp.array([[1, 0], [0, exp]])
    ptm = unitary_to_ptm(unitary_op, basis, basis)
    return ptm


def get_cz_rotation(angle: Float, basis: Basis) -> Array:
    """
    get_cphase Returns a controlled phase operation by a given angle.

    Parameters
    ----------
    angle : Float
        The controlled phase angle.

    Returns
    -------
    Array
        The controlled phase Pauli transfer matrix.
    """
    if len(basis) != 2:
        raise ValueError("CPhase gate requires two qubit bases.")

    exp = jnp.exp(1j * angle)
    diag_vals = jnp.array([1, 1, 1, exp])
    unitary_op = jnp.diag(diag_vals)
    ptm = unitary_to_ptm(unitary_op, basis, basis)
    return ptm


def get_cx_rotation(angle: Float, basis: Basis) -> Array:
    """
    get_cx_rotation Returns a controlled x-rotation operation by a given angle.

    Parameters
    ----------
    angle : Float
        The controlled x-rotation angle.

    Returns
    -------
    Array
        The controlled x-rotation Pauli transfer matrix.
    """
    if len(basis) != 2:
        raise ValueError("CX gate requires two qubit bases.")

    isin = 1j * jnp.sin(0.5 * angle)
    cos = jnp.cos(0.5 * angle)
    cx_unitary = jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos, -isin],
            [0, 0, -isin, cos],
        ]
    )
    ptm = unitary_to_ptm(cx_unitary, basis, basis)
    return ptm


def get_cy_rotation(angle: Float, basis: Basis) -> Array:
    """
    get_cy_rotation Returns a controlled y-rotation operation by a given angle.

    Parameters
    ----------
    angle : Float
        The controlled y-rotation angle.

    Returns
    -------
    Array
        The controlled y-rotation Pauli transfer matrix.
    """
    if len(basis) != 2:
        raise ValueError("CY gate requires two qubit bases.")

    sin = jnp.sin(0.5 * angle)
    cos = jnp.cos(0.5 * angle)
    cy_unitary = jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos, -sin],
            [0, 0, sin, cos],
        ]
    )
    ptm = unitary_to_ptm(cy_unitary, basis, basis)
    return ptm
