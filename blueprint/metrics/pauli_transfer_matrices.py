import jax.numpy as jnp
import math
from jax.typing import DTypeLike

"""
This file contains pauli transfer matrices for all basic qubit operations.
Most of the code is adapted from https://github.com/QudevETH/PycQED_py3/blob/qudev_master/pycqed/simulations/pauli_transfer_matrices.py
"""

DTYPE = jnp.complex128

I = jnp.eye(4)
# Pauli group
X = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=DTYPE)

Y = jnp.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=DTYPE)

Z = jnp.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=DTYPE)

# Exchange group
S = jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=DTYPE)
S2 = jnp.dot(S, S)
# Hadamard group
H = jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=DTYPE)

CZ = jnp.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=DTYPE,
)


def convert_to_rad(theta: float, unit: str = "rad"):
    if unit == "rad":
        return theta
    if unit == "deg":
        return math.degrees(theta)
    raise ValueError(f"Expected unit to be rad or deg but got {unit = }.")


def X_theta(theta: float, unit: str = "rad", dtype: DTypeLike = DTYPE):
    """
    PTM of rotation of theta along the X axis
    """
    theta = convert_to_rad(theta, unit)

    X = jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, math.cos(theta), -math.sin(theta)],
            [0, 0, math.sin(theta), math.cos(theta)],
        ],
        dtype=dtype,
    )
    return X


def Y_theta(theta: float, unit: str = "rad", dtype: DTypeLike = DTYPE):
    """
    PTM of rotation of theta along the Y axis
    """
    theta = convert_to_rad(theta, unit)

    Y = jnp.array(
        [
            [1, 0, 0, 0],
            [0, math.cos(theta), 0, math.sin(theta)],
            [0, 0, 1, 0],
            [0, -math.sin(theta), 0, math.cos(theta)],
        ],
        dtype=dtype,
    )
    return Y


def Z_theta(theta: float, unit: str = "rad", dtype: DTypeLike = DTYPE):
    """
    PTM of rotation of theta along the Z axis
    """
    theta = convert_to_rad(theta, unit)

    Z = jnp.array(
        [
            [1, 0, 0, 0],
            [0, math.cos(theta), -math.sin(theta), 0],
            [0, math.sin(theta), math.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    return Y
