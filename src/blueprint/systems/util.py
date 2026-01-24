from abc import abstractmethod
from operator import mul
from functools import reduce
from typing import Tuple

from jax import numpy as jnp
from jaxtyping import Array

from equinox import Module


class Embedding(Module):
    """Class implementing embedding utilities for quantum systems."""

    _left_id: Array
    _right_id: Array

    def __init__(self, ind: int, dims: Tuple[int, ...]) -> None:
        next_ind = ind + 1
        left_dim = reduce(mul, dims[:ind], 1)
        right_dim = reduce(mul, dims[next_ind:], 1)

        self._left_id = jnp.identity(left_dim)
        self._right_id = jnp.identity(right_dim)

    def __call__(self, op: Array) -> Array:
        """
        __call__ Embed an operator into the larger Hilbert space.

        Parameters
        ----------
        operator : Array
            Operator to be embedded.

        Returns
        -------
        Array
            Embedded operator.
        """
        embedded_op = jnp.kron(self._left_id, jnp.kron(op, self._right_id))
        return embedded_op
