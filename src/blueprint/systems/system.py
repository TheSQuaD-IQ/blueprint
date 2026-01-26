from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, Dict, Iterator, Iterable, TYPE_CHECKING

from jax import numpy as jnp
from jaxtyping import Array, Scalar
from equinox import field, Module
from dynamiqs.time_qarray import constant, TimeQArray, SummedTimeQArray

from .util import Embedding

if TYPE_CHECKING:
    from ..drives import BaseDrive as Drive


class System(Module):
    """AbstractSystem is the base class for all quantum systems in the package."""

    _label: str = field(static=True)
    _dim: int = field(static=True)

    _drives: Dict[str, Drive]

    _embedding: Embedding | None
    _ind: int | None = field(static=True)
    _dims: tuple[int, ...] | None = field(static=True)

    def __init__(
        self,
        label: str,
        dim: int,
        device_ind: int | None = None,
        device_dims: Iterable[int] | None = None,
    ) -> None:
        if not isinstance(label, str):
            raise TypeError("label must be a string.")
        self._label = label

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer.")
        self._dim = dim
        self._drives = {}

        if device_ind is None:
            if device_dims is not None:
                raise ValueError(
                    "To embed a system, both device_ind and device_dims must be provided."
                )
            self._embedding = None
        else:
            if device_dims is None:
                raise ValueError(
                    "To embed a system, both device_ind and device_dims must be provided."
                )
            if not isinstance(device_ind, int):
                raise TypeError("device_ind must be an integer.")

            try:
                device_dims = tuple(device_dims)
            except TypeError:
                raise TypeError("device_dims must be an iterable of integers.")

            for dim in device_dims:
                if not isinstance(dim, int) or dim <= 0:
                    raise ValueError(
                        "All entries in device_dims must be positive integers."
                    )
            if not 0 <= device_ind < len(device_dims):
                raise ValueError("device_ind must be a valid index for device_dims.")

            expected_dim = device_dims[device_ind]
            if expected_dim != self._dim:
                raise ValueError(
                    f"The system dim ({self._dim}) must match the corresponding dims ({expected_dim}) entry."
                )

            self._embedding = Embedding(device_ind, device_dims)
        self._ind = device_ind
        self._dims = device_dims

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

    @property
    def dim(self) -> int:
        """
        dim Returns the Hilbert space dimension of the quantum system.

        Returns
        -------
        int
            The Hilbert space dimension of the quantum system.
        """
        return self._dim

    @property
    def device_ind(self) -> int | None:
        """
        device_ind Returns the index of the system in the device Hilbert space.

        Returns
        -------
        int | None
            Index of the system in the device.
        """
        return self._ind

    @property
    def device_dims(self) -> Tuple[int, ...] | None:
        """
        device_dims Returns the dimensions of each subsystem in the device Hilbert space.

        Returns
        -------
        Tuple[int, ...] | None
            Dimensions of each subsystem in the device.
        """
        return self._dims

    @property
    def is_embedded(self) -> bool:
        """
        is_embedded Whether the quantum system has been embedded in a larger Hilbert space.

        Returns
        -------
        bool
            True if the system has a device embedding index.
        """
        return self._embedding is not None

    @property
    def drives(self) -> Iterator[Drive]:
        """
        drives Iterator over drives attached to the system.

        Returns
        -------
        Iterator[Drive]
            Iterator over drive instances.
        """
        return iter(self._drives.values())

    @property
    def drive_labels(self) -> Tuple[str, ...]:
        """
        drive_labels Labels of drives attached to the system.

        Returns
        -------
        Tuple[str, ...]
            Tuple of drive labels.
        """
        return tuple(self._drives.keys())

    @property
    def num_drives(self) -> int:
        """
        num_drives Number of drives attached to the system.

        Returns
        -------
        int
            Count of drives.
        """
        return len(self._drives)

    @property
    def is_driven(self) -> bool:
        """
        is_driven Whether the system has any drives attached.

        Returns
        -------
        bool
            True if one or more drives are present.
        """
        return any(self._drives)

    @abstractmethod
    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> System:
        """
        embed Embed the system into a larger device Hilbert space.

        Parameters
        ----------
        device_ind : int
            Index of this system in the device ordering.
        device_dims : Tuple[int, ...]
            Dimensions of each subsystem in the device.

        Returns
        -------
        System
            New system instance with embedding applied.
        """

    @abstractmethod
    def get_hamiltonian(self) -> Array:
        """
        get_hamiltonian Return the system Hamiltonian in the current basis/embedding.

        Notes
        -----
        Implementations may return diagonalized or truncated Hamiltonians
        depending on system configuration.

        Returns
        -------
        Array
            Hamiltonian matrix for the system.
        """

    def embed_op(self, operator: Array) -> Array:
        """
        embed_op Embed an operator into the device Hilbert space if needed.

        Parameters
        ----------
        operator : Array
            Operator to embed.

        Returns
        -------
        Array
            Embedded operator or original operator if no embedding is set.
        """
        if self._embedding:
            return self._embedding(operator)
        return operator

    def get_drive_hamiltonian(self, time: Scalar) -> Array:
        """get_drive_hamiltonian Return the system Hamiltonian including drives evaluated at `time`.

        Parameters
        ----------
        time : Scalar
            Time at which to evaluate drives.

        Returns
        -------
        Array
            Time-dependent Hamiltonian matrix including drives.
        """
        hamiltonian_shape = (self.dim, self.dim)
        drive_hamiltonian = jnp.zeros(hamiltonian_shape)

        for drive in self._drives.values():
            drive_hamiltonian += drive.get_hamiltonian(self, time)
        return drive_hamiltonian

    @abstractmethod
    def get_eigenvalues(self) -> Array:
        """
        get_eigenvalues Return eigenvalues of the system Hamiltonian.

        Returns
        -------
        Array
            Eigenvalues of the Hamiltonian.
        """

    @abstractmethod
    def get_eigenstates(self) -> Tuple[Array, Array]:
        """
        get_eigenstates Return eigenvalues and eigenvectors of the system Hamiltonian.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of (eigenvalues, eigenvectors).
        """

    def get_transition_op(self, start_ind: int, end_ind: int) -> Array:
        """
        get_transition_op Return an outer-product transition operator between two levels.

        Parameters
        ----------
        start_ind, end_ind : int
            Level indices for transition ``|end><start|``.

        Returns
        -------
        Array
            Transition operator embedded to device dimension.
        """
        _, eig_states = self.get_eigenstates()
        start_state = eig_states[:, start_ind]
        end_state = eig_states[:, end_ind]
        transition_op = jnp.outer(end_state, start_state)
        return self.embed_op(transition_op)

    def get_energy_diff(self, level: int, other_level: int) -> Array:
        """
        get_energy_diff Return the energy difference between two eigenlevels.

        Parameters
        ----------
        level : int
            First energy level index.
        other_level : int
            Second energy level index.

        Returns
        -------
        Array
            Energy difference ``E[level] - E[other_level]``.

        Raises
        ------
        TypeError
            If provided level indices are not integers.
        ValueError
            If indices are out of range for the system dimension.
        """
        if not isinstance(level, int):
            raise TypeError("level must be an integer.")

        if not 0 <= level < self.dim:
            raise ValueError(
                f"level must be between 0 and {self.dim}, instead got {level}."
            )

        if not isinstance(other_level, int):
            raise TypeError("other_level must be an integer.")

        if not 0 <= other_level < self.dim:
            raise ValueError(
                f"other_level must be between 0 and {self.dim}, instead got {other_level}."
            )

        energies = self.get_eigenvalues()
        energy_diff = energies[level] - energies[other_level]
        return energy_diff

    @property
    def anharmonicity(self) -> Array:
        """
        anharmonicity Returns the anharmonicity of the quantum system.

        Returns
        -------
        Array
            The anharmonicity of the quantum system.
        """
        comp_energy = self.get_energy_diff(1, 0)
        exc_energy = self.get_energy_diff(2, 1)
        anharmonicity = exc_energy - comp_energy
        return anharmonicity

    @property
    def frequency(self) -> Array:
        """
        frequency Returns the fundamental frequency of the quantum system.

        Returns
        -------
        Array
            The fundamental frequency of the quantum system
        """
        comp_energy = self.get_energy_diff(1, 0)
        return comp_energy

    def get_hamiltonian_qarray(self) -> TimeQArray:
        """
        get_hamiltonian_qarray Return the Hamiltonian wrapped as a constant TimeQArray.

        Returns
        -------
        TimeQArray
            Constant time-dependent representation of the Hamiltonian.
        """
        hamiltonian = self.get_hamiltonian()
        return constant(hamiltonian)

    def get_drive_qarray(self) -> TimeQArray:
        """
        get_drive_qarray Return summed TimeQArray of the system's drive Hamiltonians.

        Returns
        -------
        TimeQArray
            Time-dependent Hamiltonian for drives as a TimeQArray.
        """
        if not self.is_driven:
            raise ValueError("The quantum system is not driven.")

        time_arrays = [drive.get_hamiltonian_qarray(self) for drive in self.drives]

        return SummedTimeQArray(time_arrays)
