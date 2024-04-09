from __future__ import annotations
import math
from typing import Tuple

from jax import Array
from jax import numpy as jnp


class OperatorBasis:
    def __init__(self, operators: Array, labels: Tuple[str, ...]) -> None:
        """
        __init__ Initializes the OperatorBasis class.

        Parameters
        ----------
        operators : Array
            The operators that form the basis.
        labels : Tuple[str, ...]
            The labels for the operators.

        Raises
        ------
        ValueError
            If the operators is not a 3D array.
        ValueError
            If each operator is not square matrices.
        ValueError
            If the number of operators is greater than the Hilbert space dimension squared.
        ValueError
            If the number of labels does not match the number of operators.
        """
        num_dims = len(operators.shape)
        if num_dims != 3:
            raise ValueError(
                f"Operator basis tensors must have shape (pauli_dim, dim, dim), got shape {operators.shape}."
            )
        pauli_dim, dim, other_dim = operators.shape
        if dim != other_dim:
            raise ValueError(
                f"Operator basis tensors must be square matrices, got shape ({dim}, {other_dim})."
            )
        if pauli_dim > dim**2:
            raise ValueError(
                "Number of operators must be less than or equal to the Hilbert space dimension squared."
            )

        num_labels = len(labels)
        if pauli_dim != num_labels:
            raise ValueError(
                f"Number of labels must match the number of operators, got {pauli_dim} operators and {len(labels)} labels."
            )

        self._pauli_dim = pauli_dim
        self._dim = dim

        self._ops = operators
        self._labels = labels

    @property
    def dim(self) -> int:
        """
        dim Returns the dimension of the Hilbert space.

        Returns
        -------
        int
            The dimension of the Hilbert space.
        """
        return self._dim

    @property
    def pauli_dim(self) -> int:
        """
        pauli_dim Returns the Pauli dimension of the operator basis. This is the number of operators in the basis.

        Returns
        -------
        int
            The Pauli dimension of the operator basis.
        """
        return self._pauli_dim

    @property
    def operators(self) -> Array:
        return self._ops

    @property
    def labels(self) -> Tuple[str, ...]:
        return self._labels

    def subbasis(self, inds: Tuple[int, ...]) -> OperatorBasis:
        """
        subbasis Returns a subbasis of the operator basis, based on the provided indices.

        Parameters
        ----------
        inds : Tuple[int, ...]
            The indices of the operators to include in the subbasis.

        Returns
        -------
        OperatorBasis
            The subbasis of the operator basis.
        """
        ind_arr = jnp.array(inds)
        sub_ops = jnp.take(self._ops, ind_arr, axis=0)
        sub_labels = tuple(self._labels[ind] for ind in inds)
        return OperatorBasis(sub_ops, sub_labels)

    def expand_dim(self, dim: int) -> OperatorBasis:
        """
        expand_dim Expands the operator basis to the specified dimension.

        Parameters
        ----------
        dim : int
            The new dimension of the operator basis.

        Returns
        -------
        OperatorBasis
            The expanded operator basis.

        Raises
        ------
        ValueError
            If the new dimension is less than or equal to the current dimension.
        """
        if dim <= self._dim:
            raise ValueError(
                "New dimension must be greater than the current dimension."
            )

        diff = dim - self._dim
        pad_widths = ((0, 0), (0, diff), (0, diff))
        expanded_ops = jnp.pad(self._ops, pad_widths)
        return OperatorBasis(expanded_ops, self._labels)

    def truncate_dim(self, dim: int) -> OperatorBasis:
        """
        truncate_dim Truncates the operator basis to the specified dimension.

        Parameters
        ----------
        dim : int
            The new dimension of the operator basis.

        Returns
        -------
        OperatorBasis
            The truncated operator basis.

        Raises
        ------
        ValueError
            If the new dimension is greater than or equal to the current dimension.
        """
        if dim >= self._dim:
            raise ValueError("New dimension must be less than the current dimension.")
        trunc_ops = self._ops[:, :dim, :dim]
        return OperatorBasis(trunc_ops, self._labels)

    def is_orthogonal(self) -> bool:
        """
        is_orthogonal Returns whether the operator basis is orthogonal.

        Returns
        -------
        bool
            Whether the operator basis is orthogonal.
        """
        inner_prod = jnp.einsum("xij, yij -> xy", self._ops, self._ops)
        diag = jnp.diagonal(inner_prod)
        offdiag_mat = inner_prod - jnp.diag(diag)
        num_elems = int(jnp.count_nonzero(offdiag_mat))
        result = math.isclose(num_elems, 0)
        return result

    def is_orthonormal(self) -> bool:
        """
        is_orthonormal Returns whether the operator basis is orthonormal.

        Returns
        -------
        bool
            Whether the operator basis is orthonormal.
        """
        inner_prod = jnp.einsum("xij, yji -> xy", self._ops, self._ops)
        identity = jnp.identity(self.pauli_dim)
        result = bool(jnp.allclose(inner_prod, identity))
        return result

    def transform(self, trans_op: Array) -> OperatorBasis:
        """
        transform Transforms the operator basis using the provided unitary transformation operator.

        Parameters
        ----------
        trans_op : Array
            The transformation operator.

        Returns
        -------
        OperatorBasis
            The transformed operator basis.
        """
        transformed_ops = jnp.einsum(
            "ji, ajk, kl -> ail", jnp.conj(trans_op), self._ops, trans_op
        )
        return OperatorBasis(transformed_ops, self._labels)
