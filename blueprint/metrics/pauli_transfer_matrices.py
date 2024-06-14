import jax.numpy as jnp
import math

"""
This file contains pauli transfer matrices for all basic qubit operations.
Code adapted from https://github.com/QudevETH/PycQED_py3/blob/qudev_master/pycqed/simulations/pauli_transfer_matrices.py
"""

I = jnp.eye(4)
# Pauli group
X = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

Y = jnp.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

Z = jnp.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# Exchange group
S = jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
S2 = jnp.dot(S, S)
# Hadamard group
H = jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]])

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
    ]
)


def convert_to_rad(theta: float, unit: str = "rad"):
    if unit == "rad":
        return theta
    if unit == "deg":
        return math.degrees(theta)
    raise ValueError(f"Expected unit to be rad or deg but got {unit = }.")


def X_theta(theta: float, unit: str = "rad"):
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
        ]
    )
    return X


def Y_theta(theta: float, unit: str = "rad"):
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
        ]
    )
    return Y


def Z_theta(theta: float, unit: str = "rad"):
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
        ]
    )
    return Z
