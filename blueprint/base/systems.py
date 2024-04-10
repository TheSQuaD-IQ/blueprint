from abc import ABCMeta, abstractmethod
from typing import Tuple, Dict, Iterator, Callable

from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp

from ..drives import Drive
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

        self._truncated: bool = False
        self._trunc_dim: int | None = None

        self._embedded: bool = False
        self._ind: int | None = None
        self._device_dims: Tuple[int, ...] | None = None

        self._diagonalized: bool = False
        self._transform: Array | None = None

        self._drives: dict[str, Drive] = {}

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
    def num_drives(self) -> int:
        """
        num_drives Returns the number of drives applied to the transmon.

        Returns
        -------
        int
            The number of drives applied to the transmon.
        """
        return len(self._drives)

    @property
    def drives(self) -> Dict[str, Drive]:
        """
        drives Returns the dictionary of drive labels and corresponding drives that have been applied to the quantum system.

        Returns
        -------
        Dict[str, Drive]
            The dictionary of drive labels and corresponding to each applied drive.
        """
        return self._drives

    @property
    def is_truncated(self) -> bool:
        """
        is_truncated Returns whether the quantum system has been truncated.

        Returns
        -------
        bool
            Whether the quantum system has been truncated.
        """
        return self._truncated

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
    def is_driven(self) -> bool:
        """
        is_driven Returns whether the transmon is driven.

        Returns
        -------
        bool
            True if the transmon is driven, False otherwise.
        """
        return any(self._drives)

    @property
    def truncated_dim(self) -> int | None:
        """
        truncated_dim Returns the dimension of the truncated Hilbert space.

        Returns
        -------
        int
            The dimension of the truncated Hilbert space. If None, the Hilbert space is not truncated.
        """
        return self._trunc_dim

    @truncated_dim.setter
    def truncated_dim(self, dim: int | None) -> None:
        """
        truncated_dim Sets the dimension of the truncated Hilbert space.

        Parameters
        ----------
        dim : int | None
            The dimension of the truncated Hilbert space. If None, the truncation is removed.
        """
        if dim is not None:
            if not isinstance(dim, int):
                raise ValueError(
                    "The dimension of the Hilbert space must be an integer."
                )
            if dim <= 0 or dim > self._dim:
                raise ValueError(
                    f"The Hilbert space dimension ('truncated_dim') must be greater than 0 and less than or equal to the current dimension ({self._dim})."
                )
            if dim == self._dim:
                self._truncated = False
            else:
                self._truncated = True
        else:
            self._truncated = False
        self._trunc_dim = dim

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

    def _get_drive_hamiltonian(self, **params) -> Array:
        """
        _get_drive_hamiltonian Returns the sum of the Hamiltonian of each of the drives applied to the system.

        Returns
        -------
        Array
            The total drive Hamiltonian.
        """
        hamiltonian = jnp.zeros((self._dim, self._dim))

        for drive in self._drives.values():
            drive_hamiltonian = drive.get_hamiltonian(**params)

            hamiltonian = jnp.add(hamiltonian, drive_hamiltonian)
        return hamiltonian

    def get_drive_hamiltonian_terms(self, **params) -> Iterator[Tuple[Callable, Array]]:
        """
        get_drive_hamiltonian_terms

        Returns
        -------
        Array
            The total drive Hamiltonian.
        """
        for drive in self._drives.values():
            for prefactor, op in drive.decompose(**params):
                yield prefactor, self.process_op(op)

    def get_drive_hamiltonian(self, **params) -> Array:
        """
        get_drive_hamiltonian Returns the sum of the Hamiltonian of each of the drives applied to the system.

        Returns
        -------
        Array
            The total drive Hamiltonian.
        """
        drive_hamiltonian = self._get_drive_hamiltonian(**params)
        return self.process_op(drive_hamiltonian)

    def _get_diagonal_hamiltonian(self) -> Array:
        dim = self._trunc_dim or self._dim

        eig_vals = self.get_eigenvalues()
        hamiltonian = jnp.diag(eig_vals[:dim])
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
            hamiltonian = self.process_op(
                diag_hamiltonian, diagonalize=False, truncate=False
            )
            return hamiltonian

        native_hamiltonian = self._get_hamiltonian()
        hamiltonian = self.process_op(native_hamiltonian)
        return hamiltonian

    @abstractmethod
    def get_charge_op(self) -> Array:
        """
        get_charge_op Returns the charge operator of the quantum system.

        Returns
        -------
        Array
            The charge operator of the quantum system.
        """

    def _get_eigenvalues(self) -> Array:
        hamiltonian = self.get_hamiltonian()
        eig_vals = jsp.linalg.eigh(hamiltonian, eigvals_only=True)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals

    def get_eigenvalues(self, *, truncate: bool = True) -> Array:
        """
        eig_vals Returns the eigenvalues of the quantum system Hamiltonian.

        Returns
        -------
        Array
            The eigenvalues of the quantum system Hamiltonian.
        """
        eig_vals = self._get_eigenvalues()
        if truncate:
            dim = self._trunc_dim or self._dim
            return eig_vals[:dim]
        return eig_vals

    def _get_eigenstates(self) -> Tuple[Array, Array]:
        hamiltonian = self._get_hamiltonian()
        eig_vals, eig_vecs = jsp.linalg.eigh(hamiltonian, eigvals_only=False)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals, eig_vecs

    def get_eigenstates(
        self, *, diagonalize: bool = True, truncate: bool = True
    ) -> Tuple[Array, Array]:
        """
        get_eigenstates Returns the eigenvalues and eigenvectors of the quantum system Hamiltonian.

        Parameters
        ----------
        diagonalize : bool, optional
            Whether to return the diagonalized eigenstates, by default True

        Returns
        -------
        Tuple[Array, Array]
            The eigenvalues and eigenvectors of the quantum system Hamiltonian.
        """
        if truncate:
            dim = self._trunc_dim or self._dim
        else:
            dim = self._dim

        if diagonalize and self._diagonalized:
            eig_vals = self.get_eigenvalues()
            eig_vecs = jnp.identity(dim)
            return eig_vals[:dim], eig_vecs

        eig_vals, eig_vecs = self._get_eigenstates()
        return eig_vals[:dim], eig_vecs[:, :dim]

    def diagonalize(self, truncated_dim: int | None = None) -> None:
        """
        diagonalize Diagonalizes the quantum system Hamiltonian
        and truncates the Hilbert space to the specified dimension.

        Parameters
        ----------
        truncated_dim : int
            The dimension of the truncated Hilbert space.

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
            raise RuntimeError(
                f"QuantumSystem '{self._label}' has already been diagonalized."
            )

        if truncated_dim is not None:
            self.truncated_dim = truncated_dim

        _, eig_vecs = self.get_eigenstates(truncate=False)

        self._diagonalized = True
        self._transform = eig_vecs

    def embed(self, ind: int, device_dims: Tuple[int, ...]) -> None:
        """
        embed Embeds the quantum system into a larger Hilbert space.

        Parameters
        ----------
        ind : int
            The index of the quantum system in the larger Hilbert space.
        device_dims : Tuple[int]
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
            raise RuntimeError(
                f"QuantumSystem '{self._label}' has already been embedded. Create a new QuantumSystem object to embed into another system."
            )

        num_qubits = len(device_dims)
        if not 0 <= ind < num_qubits:
            raise ValueError(
                f"The index of the quantum system ({ind}) must be between 0 and the number of quantum systems in the system ({num_qubits})."
            )

        dim = self._trunc_dim or self._dim

        if device_dims[ind] != dim:
            raise ValueError(
                f"The Hilbert dimension of the quantum system ({dim}) must match the dimension of the Hilbert subspace it is being embedded into ({device_dims[ind]})."
            )

        self._embedded = True
        self._ind = ind
        self._device_dims = device_dims

    def process_op(
        self,
        op: Array,
        *,
        diagonalize: bool = True,
        truncate: bool = True,
        embed: bool = True,
    ) -> Array:
        """
        process_op Processes an operator expressed in the default quantum system basis to another basis.
        This can include both transformation to the energy basis if the qubit has been diagonalized,
        a truncation of the operator if the dimension of the hilbert space has been truncated,
        and an embedding of the operator in a higher-dimensional Hilbert space if the qubit has been
        included in a device.

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
        if diagonalize and self._diagonalized:
            # Handle the case where the qubit implements the operators in an already diagonalized basis.
            if self._transform is None:
                raise ValueError(
                    "The transform matrix is not set, making it impossible to perform this."
                )
            op = transform_op(op, self._transform)

        if truncate and self._truncated:
            if self._trunc_dim is None:
                raise ValueError(
                    "The truncation dimension is not set, making it impossible to perform this."
                )
            op = op[: self._trunc_dim, : self._trunc_dim]

        if embed and self._embedded:
            if self._ind is None:
                raise ValueError(
                    "The embedding index is not set, making it impossible to perform this."
                )
            if self._device_dims is None:
                raise ValueError(
                    "The device dimensions are not set, making it impossible to perform this."
                )
            op = embed_op(op, self._ind, self._device_dims)
        return op

    def get_freq_difference(self, low_ind: int, high_ind: int) -> Array:
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

        eig_vals = self.get_eigenvalues()
        freq_diff = eig_vals[high_ind] - eig_vals[low_ind]
        return freq_diff

    @property
    def anharmonicity(self) -> Array:
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
    def frequency(self) -> Array:
        """
        frequency Returns the fundamental frequency of the qubit.

        Returns
        -------
        float
            The fundamental frequency of the qubit.
        """
        frequency = self.get_freq_difference(0, 1)
        return frequency

    def get_eigenstate(self, state_index: int) -> Tuple[Array, Array]:
        """
        get_eigenstate Returns the eigen energy and eigen state of a specific state, as specified by an integer state index, as the determined by the energy of that state.

        Parameters
        ----------
        state_index : int
            The index of the state, as determined by the energy/number of excitations. For example, the ground state has an index of 0. The second-excitation state has an index of 2.

        Returns
        -------
        Tuple[Array, Array]
            The eigen energy and eigen state of the specified state.
        """
        # NOTE: This function is very trivial and mostly added for consistency between device and qubits. Should we get rid of it?
        eig_vals, eig_vecs = self.get_eigenstates()
        energy = eig_vals[state_index]
        state = eig_vecs[state_index]
        return energy, state

    def _get_comp_projector(self) -> Array:
        _, eig_vecs = self._get_eigenstates()
        comp_states = eig_vecs[:, :2]
        projector = comp_states @ jnp.conj(comp_states.T)
        return projector

    def get_comp_projector(self) -> Array:
        """
        get_comp_projector Returns the projector onto the computational subspace.

        Returns
        -------
        Array
            The projector onto the computational subspace.
        """
        if self._diagonalized:
            comp_elems = jnp.ones(2)
            pad_widths = (0, self._dim - 2)
            diag_elems = jnp.pad(comp_elems, pad_widths)
            projector = jnp.diag(diag_elems)
            return self.process_op(projector, diagonalize=False)

        comp_projector = self._get_comp_projector()
        return self.process_op(comp_projector)

    def get_leak_projector(self) -> Array:
        """
        get_leak_projector Returns the projector onto the leakage subspace.

        Returns
        -------
        Array
            The projector onto the leakage subspace.
        """
        if self._diagonalized:
            comp_elems = jnp.zeros(2)
            pad_widths = (0, self._dim - 2)
            diag_elems = jnp.pad(comp_elems, pad_widths, constant_values=1)
            projector = jnp.diag(diag_elems)
            return self.process_op(projector, diagonalize=False)

        id_op = jnp.identity(self._dim)
        comp_projector = self._get_comp_projector()
        leak_projector = id_op - comp_projector
        return self.process_op(leak_projector)


def get_pos_eigenvectors(eig_vecs: Array) -> Array:
    vecs = eig_vecs.T
    for ind, vec in enumerate(vecs):
        vec_ind = jnp.argmax(abs(vec))
        angle = jnp.angle(vec[vec_ind])
        phase = jnp.exp(-1j * angle)
        vecs.at[ind].set(phase * vec)
    return jnp.transpose(vecs)
