from abc import ABCMeta, abstractmethod
from typing import Tuple

from jax import numpy as jnp
from jax import scipy as jsp
from jax import Array

from ..util.linalg import embed_op, transform_op


class QuantumSystem(metaclass=ABCMeta):
    """
    Base quantum system class.
    """

    def __init__(self, label: str, dim: int) -> None:
        if not isinstance(label, str):
            raise ValueError(
                f"The label of the quantum system must be a string, instead got type {type(label)}."
            )
        self._label: str = label

        if not isinstance(dim, int):
            raise ValueError(
                f"The dimension of the Hilbert space must be an integer, instead got type {type(dim)}."
            )
        self._dim: int = dim

        self._embedded: bool = False
        self._ind: int | None = None
        self._dims: Tuple[int] | None = None

        self._diagonalized: bool = False
        self._transform: Array = None

        self._eig_vals: Array = None

    @property
    def label(self) -> str:
        """
        label Returns the label of the quantum system.

        Returns
        -------
        str
            The label of the quantum system.
        """
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        """
        label Sets the label of the quantum system.

        Parameters
        ----------
        label : str
            The new label of the system, must a string.
        """
        if not isinstance(label, str):
            raise ValueError(
                f"The label of the quantum system must be a string, instead got type {type(label)}."
            )
        self._label = label

    @property
    def is_diagonalized(self) -> bool:
        """
        is_diagonalized Returns whether the quantum system has been diagonalized.

        Returns
        -------
        bool
            Whether the quantum system has been diagonalized.
        """
        return self._diagonalized

    @property
    def is_embedded(self) -> bool:
        """
        is_embedded Returns whether the quantum system is embedded into a larger Hilbert space.

        Returns
        -------
        bool
            Whether the quantum system is embedded into a larger Hilbert space.
        """
        return self._ind is not None

    @property
    def dim(self) -> int:
        """
        hilbert_dim Returns the dimension of the Hilbert space of the quantum system.

        Returns
        -------
        int
            The dimension of the Hilbert space of the quantum system.
        """
        return self._dim

    @abstractmethod
    def _get_hamiltonian(self) -> Array:
        pass

    def _get_diagonal_hamiltonian(self, *, sub_ground_energy: bool = True) -> Array:
        if self._eig_vals is None:
            # Case where it was not diagonalized
            eig_vals = self.eigenvalues()
            diagonal = eig_vals[:self._dim]

        else:
            # Case where the Hamiltonian was diagonalized or the eigenvalues were previously computed.
            diagonal = self._eig_vals[:self._dim]
        
        if sub_ground_energy:
            diagonal = diagonal - diagonal[0]
        hamiltonian = jnp.diag(diagonal)
        return hamiltonian

    def get_hamiltonian(self) -> Array:
        """
        get_hamiltonian Returns the Hamiltonian of the quantum system.

        Returns
        -------
        Array
            The Hamiltonian of the quantum system.
        """
        if self.is_diagonalized:
            diag_hamiltonian = self._get_diagonal_hamiltonian()
            hamiltonian = self.process_op(diag_hamiltonian, diagonalize=False)
            return hamiltonian
    
        native_hamil = self._get_hamiltonian()
        hamil = self.process_op(native_hamil)
        return hamil

    def eigenvalues(self, **kwargs) -> Array:
        """
        eig_vals Returns the eigenvalues of the quantum system Hamiltonian.

        Returns
        -------
        Array
            The eigenvalues of the quantum system Hamiltonian.
        """
        hamiltonian = self._get_hamiltonian()
        eig_vals = jsp.linalg.eigh(hamiltonian, eigvals_only=True, **kwargs)
        if self._eig_vals is None:
            self._eig_vals = eig_vals
        return eig_vals

    def eigenstates(self, **kwargs) -> Tuple[Array, Array]:
        """
        eig_sys Returns the eigenvalues and eigenvectors of the quantum system Hamiltonian.

        Returns
        -------
        Tuple[Array, Array]
            The eigenvalues and eigenvectors of the quantum system Hamiltonian.
        """
        hamiltonian = self._get_hamiltonian()
        eig_vals, eig_vecs = jsp.linalg.eigh(hamiltonian, eigvals_only=False, **kwargs)
        if self._eig_vals is None:
            self._eig_vals = eig_vals
        return eig_vals, eig_vecs

    def diagonalize(
        self, truncated_dim: int | None = None, *, sub_ground_energy: bool = True
    ) -> None:
        """
        diagonalize Diagonalizes the quantum system Hamiltonian
        and truncates the Hilbert space to the specified dimension.

        Parameters
        ----------
        truncated_dim : int
            The dimension of the truncated Hilbert space.
        sub_ground_energy : bool, optional
            Whether to subtract the ground state energy from the eigenvalues, by default True

        Raises
        ------
        ValueError
            If the dimension of the Hilbert space is not an integer.
        ValueError
            If the dimension of the Hilbert space is not
            greater than 0 and less than or equal to the
            current dimension.
        """
        if self.is_diagonalized:
            raise RuntimeError(f"QuantumSystem '{self._label}' has already been diagonalized.")

        if truncated_dim is not None:
            if not isinstance(truncated_dim, int):
                raise ValueError(
                    "The dimension of the Hilbert space must be an integer."
                )
        if truncated_dim <= 0 or truncated_dim > self._dim:
            raise ValueError(
                f"The Hilbert space dimension ('truncated_dim') must be greater than 0 and less than or equal to the current dimension ({self._dim})."
            )
            
        eig_vals, eig_vecs = self.eigenstates()
        trunc_vals = eig_vals[:truncated_dim]
        if sub_ground_energy:
            trunc_vals = trunc_vals - trunc_vals[0]

        trunc_vecs = eig_vecs[:, :truncated_dim]
        trunc_vecs = get_pos_eigenvectors(trunc_vecs)

        self._diagonalized = True
        self._transform = trunc_vecs
        self._dim = truncated_dim

    def embed(self, ind: int, dims: Tuple[int]) -> None:
        """
        embed Embeds the quantum system into a larger Hilbert space.

        Parameters
        ----------
        ind : int
            The index of the quantum system in the larger Hilbert space.
        dims : Tuple[int]
            The dimension of each quantum system (including this one) in the full Hilbert space.

        Raises
        ------
        ValueError
            If the index of the quantum system is not between 0 and the total number of quantum systems in the system.
        ValueError
            If the dimension of the Hilbert space of the quantum system
            does not match the dimension of the Hilbert
            subspace it is being embedded into.
        """
        if self.is_embedded:
            raise RuntimeError(f"QuantumSystem '{self._label}' has already been embedded. Create a new QuantumSystem object to embed into another system.")

        num_systems = len(dims)
        if not 0 <= ind < num_systems:
            raise ValueError(
                f"The index of the quantum system ({ind}) must be between 0 and the number of quantum systems in the system ({num_systems})."
            )

        if dims[ind] != self._dim:
            raise ValueError(
                f"The Hilbert dimension of the quantum system ({self._dim}) must match the dimension of the Hilbert subspace it is being embedded into ({dims[ind]})."
            )

        self._embedded = True
        self._ind = ind
        self._dims = dims

    def process_op(
        self, op: Array, *, diagonalize: bool = True, embed: bool = True
    ) -> Array:
        """
        process_op Processes an operator in the native quantum system basis to the transformed basis.
        This can include both transformation to the diagonalized basis and/or
        an embedding of the operator in a higher-dimensional Hilbert space.

        Parameters
        ----------
        op : Array
            The operator expressed in the native basis used by the quantum system.
        diagonalize : bool, optional
            Whether to diagonalize the operator, by default True
        embed : bool, optional
            Whether to embed the operator in a higer-dimensional Hilbert space, by default True

        Returns
        -------
        Array
            _description_
        """
        if self.is_diagonalized and diagonalize:
            # Handle the case where the qubit implements the operators in an already diagonalized basis.
            if self._transform is not None: 
                op = transform_op(op, self._transform)

        if self.is_embedded and embed:
            op = embed_op(op, self._ind, self._dims)

        return op

    def get_freq_difference(self, low_ind: int, high_ind: int) -> float:
        """
        get_freq_difference Returns the frequency difference between two energy levels.

        Parameters
        ----------
        low_ind : int
            The index of the lower energy level.
        high_ind : int
            The index of the higher energy level.

        Returns
        -------
        float
            The frequency difference between the two energy levels.

        Raises
        ------
        ValueError

        ValueError
            If any of the two indices are not integers.
        ValueError
            If any of the two indices are not between 0 and the dimension of the Hilbert space.
        """
        inds = (low_ind, high_ind)

        for ind in inds:
            if not isinstance(ind, int):
                raise ValueError(
                    f"The indices of the energy levels must be integers, instead got type {type(ind)}."
                )
            if not 0 <= ind < self._dim:
                raise ValueError(
                    f"The indices of the energy levels must be between 0 and the dimension of the Hilbert space ({self._dim})."
                )

        eig_vals = self.eigenvalues()
        freq_diff = eig_vals[high_ind] - eig_vals[low_ind]
        return freq_diff

    @property
    def abs_anharmonicity(self) -> float:
        """
        anharmonicity Returns the anharmonicity of the qubit.

        Returns
        -------
        float
            The anharmonicity of the qubit.
        """
        comp_freq = self.get_freq_difference(0, 1)
        exc_freq = self.get_freq_difference(1, 2)

        anharmonicity = exc_freq - comp_freq
        return anharmonicity

    @property
    def fundamental_frequency(self) -> float:
        """
        fundamental_freq Returns the fundamental frequency of the qubit.

        Returns
        -------
        float
            The fundamental frequency of the qubit.
        """
        fundamental_freq = self.get_freq_difference(0, 1)
        return fundamental_freq


def get_pos_eigenvectors(eig_vecs: Array) -> Array:
    vecs = eig_vecs.T
    for ind, vec in enumerate(vecs):
        vec_ind = jnp.argmax(abs(vec))
        angle = jnp.angle(vec[vec_ind])
        phase = jnp.exp(-1j * angle)
        vecs.at[ind].set(phase * vec)
    return jnp.transpose(vecs)
