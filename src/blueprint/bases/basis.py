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
        dtype Data type of the basis operators.

        Returns
        -------
        jnp.dtype
            Data type of the operators array.
        """
        return self.operators.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        shape Shape of the operators array.

        Returns
        -------
        Tuple[int, ...]
            Shape of the operators array (pauli_dim, hilbert_dim, hilbert_dim).
        """
        return self.operators.shape

    def get_operator_norms(self) -> Array:
        """
        get_operator_norms Return vector of norms for each operator in the basis.

        Returns
        -------
        Array
            Norms for each basis operator (shape ``(pauli_dim,)``).
        """
        inner_prods = jnp.einsum("aij, aji -> a", self.operators, self.operators)
        operator_norms = jnp.sqrt(inner_prods)
        return operator_norms

    def normalize(self) -> Basis:
        """
        normalize Return a normalized `Basis` where each operator has unit norm.

        Returns
        -------
        Basis
            Normalized basis instance.
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
            Boolean scalar indicating whether basis operators are normalized.
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
            Boolean scalar indicating orthogonality of the basis.
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
            Boolean scalar indicating orthonormality of the basis.
        """
        inner_prods = jnp.einsum("aij, bji -> ab", self.operators, self.operators)
        expected_prods = jnp.eye(self.pauli_dim)
        return jnp.allclose(inner_prods, expected_prods)

    def to_vector(self, operator: Array) -> Array:
        """
        to_vector Project an operator into the basis to obtain its vector representation.

        Parameters
        ----------
        operator : Array
            Operator to project (shape ``(hilbert_dim, hilbert_dim)``).

        Returns
        -------
        Array
            Vector representation of the operator (shape ``(pauli_dim,)``).
        """
        vector = jnp.einsum("aij, ji -> a", self.operators, operator)
        return vector

    def to_operator(self, vector: Array) -> Array:
        """
        to_operator Reconstruct an operator from its vector representation in the basis.

        Parameters
        ----------
        vector : Array
            Vector of coefficients (shape ``(pauli_dim,)``).

        Returns
        -------
        Array
            Operator reconstructed from the basis (shape ``(hilbert_dim, hilbert_dim)``).
        """
        operator = jnp.einsum("aij, a -> ij", self.operators, vector)
        return operator

    def truncate_hilbert_dim(self, trunc_dim: int) -> Basis:
        """
        truncate_hilbert_dim Return a new `Basis` truncated to `trunc_dim` Hilbert dimension.

        Parameters
        ----------
        trunc_dim : int
            New Hilbert-space dimension (must be <= current dimension).

        Returns
        -------
        Basis
            Truncated basis instance.
        """
        if trunc_dim > self.hilbert_dim:
            raise ValueError("New dimension must be less than the current dimension.")

        trunc_operators = self.operators[  # pylint: disable=E1136
            :, :trunc_dim, :trunc_dim
        ]
        return Basis(trunc_operators, self.labels)

    def expand_hilbert_dim(self, exp_dim: int) -> Basis:
        """
        expand_hilbert_dim Return a new `Basis` expanded to `exp_dim` Hilbert dimension.

        Parameters
        ----------
        exp_dim : int
            New Hilbert-space dimension (must be > current dimension).

        Returns
        -------
        Basis
            Expanded basis instance.
        """
        if exp_dim <= self.hilbert_dim:
            raise ValueError(
                "New dimension must be greater than the current dimension."
            )

        pad_width = exp_dim - self.hilbert_dim
        pad_widths = ((0, 0), (0, pad_width), (0, pad_width))
        exp_operators = jnp.pad(self.operators, pad_widths, mode="constant")
        return Basis(exp_operators, self.labels)
