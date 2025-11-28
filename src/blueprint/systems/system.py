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
        is_embedded Returns whether the quantum system has been embedded in a larger Hilbert space.

        Returns
        -------
        bool
            Whether the quantum system has been embedded.
        """
        return self.device_ind is not None

    @property
    def drive_iter(self) -> Iterator["Drive"]:
        """
        drives Returns the drives acting on the quantum system.

        Returns
        -------
        Tuple[Drive, ...]
            The drives acting on the quantum system.
        """
        return iter(self.drives.values())

    @property
    def drive_labels(self) -> Tuple[str, ...]:
        """
        drive_labels Returns the labels of the drives acting on the quantum system.

        Returns
        -------
        Tuple[str, ...]
            The labels of the drives acting on the quantum system.
        """
        return tuple(self.drives.keys())

    @property
    def num_drives(self) -> int:
        """
        num_drives Returns the number of drives acting on the quantum system.

        Returns
        -------
        int
            The number of drives acting on the quantum system.
        """
        return len(self.drives)

    @property
    def is_driven(self) -> bool:
        """
        is_driven Returns whether the quantum system is driven.

        Returns
        -------
        bool
            Whether the quantum system is driven.
        """
        return any(self.drives)

    def add_drive(self, drive: "Drive") -> None:
        """
        add_drive Adds a drive to the quantum system.

        Parameters
        ----------
        drive : Drive
            The drive to add to the quantum system.

        Returns
        -------
        Self
            The quantum system with the drive added.
        """
        self.drives[drive.label] = drive

    @abstractmethod
    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Self:
        """
        embed Embeds the quantum system into a larger Hilbert space.

        Parameters
        ----------
        ind : int
            The index of the quantum system in the larger Hilbert space.
        device_dims : Tuple[int]
            The dimension of each quantum system (including this one) in the full device.
        """

    @abstractmethod
    def get_hamiltonian(self) -> Array:
        """
        get_hamiltonian Returns the Hamiltonian of the quantum system.
        If the system has been diagonalized, the Hamiltonian is transformed to the energy basis.
        If the system has been truncated, the Hamiltonian is truncated to the truncated dimension.
        If the system has been embedded, the Hamiltonian is embedded in the larger Hilbert space.

        Returns
        -------
        Array
            The Hamiltonian of the quantum system.
        """

    def get_drive_hamiltonian(self, time: ScalarLike) -> Array:
        """
        get_drive_hamiltonian Returns the Hamiltonian of the quantum system
        with the drives included.

        Parameters
        ----------
        time : ScalarLike
            The time at which to evaluate the drives.

        Returns
        -------
        Array
            The Hamiltonian of the quantum system with the drives included.
        """
        hamiltonian_shape = (self.dim, self.dim)
        drive_hamiltonian = jnp.zeros(hamiltonian_shape)

        for drive in self.drives.values():
            drive_hamiltonian += drive.get_hamiltonian(self, time)
        return drive_hamiltonian

    @abstractmethod
    def get_eigenvalues(self) -> Array:
        """
        _get_eigenvalues Returns the eigenvalues of the Hamiltonian of the quantum system.

        Returns
        -------
        Array
            The eigenvalues of the Hamiltonian.
        """

    @abstractmethod
    def get_eigenstates(self) -> Tuple[Array, Array]:
        """
        _get_eigenstates Returns the eigenvalues and eigenvectors
        of the Hamiltonian of the quantum system.

        Returns
        -------
        Tuple[Array, Array]
            The eigenvalues and eigenvectors of the Hamilton
        """

    def get_energy_diff(self, level: int, other_level: int) -> Array:
        """
        get_energy_diff Returns the energy difference between two levels of the quantum system.

        Parameters
        ----------
        level : int
            The level of the system for which to calculate the energy difference.
        other_level : int
            The other level of the system for which to calculate the energy difference.

        Returns
        -------
        Array
            The energy difference between the two levels.

        Raises
        ------
        TypeError
            If the level is not an integer.
        ValueError
            If the level is not between 0 and the dimension of the system.
        TypeError
            If the other_level is not an integer.
        ValueError
            If the other_level is not between 0 and the dimension of the system.
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
        get_hamiltonian_qarray Returns the Hamiltonian of the quantum system as a TimeQArray.

        Returns
        -------
        TimeQArray
            The Hamiltonian of the quantum system as a TimeArray.
        """
        hamiltonian = self.get_hamiltonian()
        return constant(hamiltonian)

    def get_drive_qarray(self) -> TimeQArray:
        """
        get_drive_qarray
        Returns the drive Hamiltonian of the quantum system as a TimeQArray.


        Returns
        -------
        TimeArray
            The Hamiltonian of the quantum system with the drives included as a TimeArray.
        """
        if not self.is_driven:
            raise ValueError("The quantum system is not driven.")

        time_arrays = [drive.get_hamiltonian_qarray(self) for drive in self.drive_iter]

        return SummedTimeQArray(time_arrays)

    def embed_op(self, operator: Array) -> Array:
        """
        embed_op Embeds the operator in a larger Hilbert space.

        Parameters
        ----------
        operator : Array
            The operator to embed.

        Returns
        -------
        Array
            The embedded operator.
        """
        if self.is_embedded:
            return embed_op(operator, self.device_ind, self.device_dims)
        return operator
