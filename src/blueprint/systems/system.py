from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, Dict, Iterator, Iterable, TYPE_CHECKING

from jax import numpy as jnp
from jaxtyping import Array, ScalarLike
from equinox import field, Module
from dynamiqs.time_qarray import constant, TimeQArray, SummedTimeQArray

from .util import Embedding, BaseEmbedding, DeviceEmbedding

if TYPE_CHECKING:
    from ..drives import BaseDrive as Drive


class System(Module):
    """AbstractSystem is the base class for all quantum systems in the package."""

    label: str = field(static=True)
    dim: int = field(static=True)

    drives: Dict[str, "Drive"]

    _embedding: Embedding
    device_ind: int | None = field(static=True)

    def __init__(
        self,
        label: str,
        dim: int,
        device_ind: int | None = None,
        device_dims: Iterable[int] | None = None,
    ) -> None:
        if not isinstance(label, str):
            raise TypeError("label must be a string.")
        self.label = label

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer.")
        self.dim = dim

        self.drives = {}

        if device_ind is None:
            if device_dims is not None:
                raise ValueError(
                    "To embed a system, both device_ind and device_dims must be provided."
                )
            self._embedding = BaseEmbedding()
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
            if expected_dim != self.dim:
                raise ValueError(
                    f"The system dim ({self.dim}) must match the corresponding device_dims ({expected_dim}) entry."
                )

            self._embedding = DeviceEmbedding(device_ind, device_dims)
        self.device_ind = device_ind

    @property
    def is_embedded(self) -> bool:
        """
        is_embedded Whether the quantum system has been embedded in a larger Hilbert space.

        Returns
        -------
        bool
            True if the system has a device embedding index.
        """
        return isinstance(self._embedding, DeviceEmbedding)

    @property
    def drive_iter(self) -> Iterator["Drive"]:
        """
        drive_iter Iterator over drives attached to the system.

        Returns
        -------
        Iterator[Drive]
            Iterator over drive instances.
        """
        return iter(self.drives.values())

    @property
    def drive_labels(self) -> Tuple[str, ...]:
        """
        drive_labelsLabels of drives attached to the system.

        Returns
        -------
        Tuple[str, ...]
            Tuple of drive labels.
        """
        return tuple(self.drives.keys())

    @property
    def num_drives(self) -> int:
        """
        num_drives Number of drives attached to the system.

        Returns
        -------
        int
            Count of drives.
        """
        return len(self.drives)

    @property
    def is_driven(self) -> bool:
        """
        is_driven Whether the system has any drives attached.

        Returns
        -------
        bool
            True if one or more drives are present.
        """
        return any(self.drives)

    def add_drive(self, drive: "Drive") -> None:
        """
        add_drive Attach a drive to the system.

        Parameters
        ----------
        drive : Drive
            Drive instance to attach.
        """
        self.drives[drive.label] = drive

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

    def get_drive_hamiltonian(self, time: ScalarLike) -> Array:
        """get_drive_hamiltonian Return the system Hamiltonian including drives evaluated at `time`.

        Parameters
        ----------
        time : ScalarLike
            Time at which to evaluate drives.

        Returns
        -------
        Array
            Time-dependent Hamiltonian matrix including drives.
        """
        hamiltonian_shape = (self.dim, self.dim)
        drive_hamiltonian = jnp.zeros(hamiltonian_shape)

        for drive in self.drives.values():
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

        time_arrays = [drive.get_hamiltonian_qarray(self) for drive in self.drive_iter]

        return SummedTimeQArray(time_arrays)

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
        return self._embedding(operator)
