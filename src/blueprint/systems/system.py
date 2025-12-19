from abc import abstractmethod
from operator import is_
from typing import Tuple, Self, Dict, Iterator, TYPE_CHECKING

from jax import numpy as jnp
from jaxtyping import Array, ScalarLike
from equinox import field, Module
from dynamiqs.time_qarray import constant, TimeQArray, SummedTimeQArray

from ..util.linalg import embed_op

if TYPE_CHECKING:
    from ..drives import BaseDrive as Drive


class System(Module):
    """AbstractSystem is the base class for all quantum systems in the package."""

    label: str = field(static=True)
    dim: int = field(static=True)

    drives: Dict[str, "Drive"]

    device_ind: int | None = field(static=True)
    device_dims: Tuple[int, ...] | None = field(static=True)

    @abstractmethod
    def __init__(self):
        pass

    def __check_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("label must be a string.")

        if not isinstance(self.dim, int):
            raise TypeError("dim must be an integer.")

        if self.dim <= 0:
            raise ValueError("dim must be a positive integer.")

        if self.device_ind is not None:
            if not isinstance(self.device_ind, int):
                raise TypeError("ind must be an integer.")

            if self.device_ind < 0:
                raise ValueError("ind must be a non-negative integer.")

            num_qubits = len(self.device_dims)
            if self.device_ind >= num_qubits:
                raise ValueError(
                    f"ind must be less than the number of qubits in the device ({num_qubits})."
                )
            if self.device_dims[self.device_ind] != self.dim:
                raise ValueError(
                    f"The Hilbert dimension of the quantum system ({self.dim}) must match the dimension of the Hilbert subspace it is being embedded into ({self.device_dims[self.device_ind]})."
                )

    @property
    def is_embedded(self) -> bool:
        """
        is_embedded Whether the quantum system has been embedded in a larger Hilbert space.

        Returns
        -------
        bool
            True if the system has a device embedding index.
        """
        return self.device_ind is not None

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
    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Self:
        """
        embed Embed the system into a larger device Hilbert space.

        Parameters
        ----------
        device_ind : int
            Index of this system in the device ordering.
        device_dims : Tuple[int, ...]
            Dimensions of each subsystem in the device.
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
    def fundamental_frequency(self) -> Array:
        """
        fundamental_frequency Returns the fundamental frequency of the quantum system.

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
        if self.is_embedded:
            return embed_op(operator, self.device_ind, self.device_dims)
        return operator
