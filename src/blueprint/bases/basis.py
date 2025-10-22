"""Module implementing a Basis class representing a complete orthogonal basis set for a quantum system."""

from __future__ import annotations

from typing import List, Tuple

from jax import numpy as jnp
from jax import Array
from equinox import Module, field


class Basis(Module):
    operators: Array
    labels: List[str] = field(static=True)

    hilbert_dim: int = field(static=True)
    pauli_dim: int = field(static=True)

    def __init__(self, operators: Array, labels: List[str]):
        pauli_dim, hilbert_dim, other_dim = operators.shape
        num_labels = len(labels)
        if pauli_dim != num_labels:
            raise ValueError("Number of labels must match the number of operators.")

        if hilbert_dim != other_dim:
            raise ValueError("Operators must be square matrices.")

        for label in labels:
            if not isinstance(label, str):
                raise TypeError("All labels must be strings.")

        self.operators = jnp.asarray(operators)
        self.labels = list(labels)

        self.hilbert_dim = int(hilbert_dim)
        self.pauli_dim = int(pauli_dim)

    def __eq__(self, other: Basis) -> bool:
        if isinstance(other, Basis):
            if self.shape == other.shape:
                if jnp.allclose(self.operators, other.operators):
                    return True
        return False

    def __len__(self):
        return self.pauli_dim

    def __repr__(self):
        class_name = self.__class__.__name__
        labels_str = " ".join(self.labels)
        repr_str = f"{class_name}, dim={self.hilbert_dim}, operators=({labels_str})"

        return repr_str

    @property
    def dtype(self) -> jnp.dtype:
        """
        dtype Returns the data type of the operators.

        Returns
        -------
        jnp.dtype
            The data type of the operators.
        """
        return self.operators.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        shape Returns the shape of the operators.

        Returns
        -------
        Tuple[int, int, int]
            The shape of the operators.
        """
        return self.operators.shape

    def get_operator_norms(self) -> Array:
        """
        get_operator_norms Returns the norms of each of the operators.

        Returns
        -------
        Array
            The norms of each of the operators.
        """
        inner_prods = jnp.einsum("aij, aji -> a", self.operators, self.operators)
        operator_norms = jnp.sqrt(inner_prods)
        return operator_norms

    def normalize(self) -> Basis:
        """
        normalize Returns a normalized operator basis.

        Returns
        -------
        Self
            A normalized operator basis.
        """
        operator_norms = self.get_operator_norms()
        normalized_operators = self.operators / operator_norms[:, None, None]
        return Basis(normalized_operators, self.labels)

    @property
    def is_normalized(self) -> Array:
        """
        is_normalized Whether the operator basis is normalized.

        Returns
        -------
        Array
            Whether the operator basis is normalized.
        """
        inner_prods = jnp.einsum("aij, aji -> a", self.operators, self.operators)
        expected_vec = jnp.ones(self.pauli_dim)
        return jnp.allclose(inner_prods, expected_vec)

    @property
    def is_orthogonal(self) -> Array:
        """
        is_orthogonal Whether the operator basis is orthogonal.

        Returns
        -------
        Array
            Whether the operator basis is orthogonal.
        """
        inner_prods = jnp.einsum("aij, bji -> ab", self.operators, self.operators)
        diag_elements = jnp.diag(inner_prods)
        diag_matrix = jnp.diag(diag_elements)
        return jnp.allclose(inner_prods, diag_matrix)

    @property
    def is_orthonormal(self) -> Array:
        """
        is_orthonormal Whether the operator basis is orthonormal.

        Returns
        -------
        Array
            Whether the operator basis is orthonormal.
        """
        inner_prods = jnp.einsum("aij, bji -> ab", self.operators, self.operators)
        expected_prods = jnp.eye(self.pauli_dim)
        return jnp.allclose(inner_prods, expected_prods)

    def to_vector(self, operator: Array) -> Array:
        """
        to_vector When given a operator representation of a state,
        returns the vector representation of the state in the operator basis.

        Parameters
        ----------
        operator : Array
            The operator to be converted to a vector.

        Returns
        -------
        Array
            The vector representation of the state in the operator basis.
        """
        vector = jnp.einsum("aij, ji -> a", self.operators, operator)
        return vector

    def to_operator(self, vector: Array) -> Array:
        """
        to_operator Converts a vector representation of a state in the operator basis,
        returns the operator representation of the state.

        Parameters
        ----------
        vector : Array
            The vector representation of the state in the operator basis.

        Returns
        -------
        Array
            The operator representation of the state.
        """
        operator = jnp.einsum("aij, a -> ij", self.operators, vector)
        return operator

    def truncate_hilbert_dim(self, trunc_dim: int) -> Basis:
        """
        truncate_hilbert_dim Truncates the operator basis to a new Hilbert space dimension.

        Parameters
        ----------
        trunc_dim : int
            The new Hilbert space dimension.

        Returns
        -------
        Basis
            The truncated operator basis.
        """
        if trunc_dim > self.hilbert_dim:
            raise ValueError("New dimension must be less than the current dimension.")

        trunc_operators = self.operators[  # pylint: disable=E1136
            :, :trunc_dim, :trunc_dim
        ]
        return Basis(trunc_operators, self.labels)

    def expand_hilbert_dim(self, exp_dim: int) -> Basis:
        """
        expand_hilbert_dim Expands the operator basis to a new Hilbert space dimension.

        Parameters
        ----------
        exp_dim : int
            The new Hilbert space dimension.

        Returns
        -------
        Basis
            The expanded operator basis.
        """
        if exp_dim <= self.hilbert_dim:
            raise ValueError(
                "New dimension must be greater than the current dimension."
            )

        pad_width = exp_dim - self.hilbert_dim
        pad_widths = ((0, 0), (0, pad_width), (0, pad_width))
        exp_operators = jnp.pad(self.operators, pad_widths, mode="constant")
        return Basis(exp_operators, self.labels)
