import math
from typing import Dict, List, Tuple, Iterable

from jax import numpy as jnp
from jax.scipy.linalg import eigh
from jaxtyping import Array, ScalarLike
from equinox import Module, field

from dynamiqs.time_qarray import constant, ConstantTimeQArray, SummedTimeQArray

from ..systems import System
from ..couplings import Coupling


class Device(Module):
    """Class implementing a quantum device."""

    _systems: Dict[str, System]
    _couplings: Dict[str, Coupling]

    _coupled_systems: Dict[str, Tuple[str, ...]] = field(static=True)

    system_dims: Tuple[int, ...] = field(static=True)
    dim: int = field(static=True)

    def __init__(self, systems: Iterable[System]):
        for system in systems:
            if system.is_embedded:
                raise ValueError("All systems must be not be embedded.")

        dims = tuple(system.dim for system in systems)
        self._systems = {
            system.label: system.embed(ind, dims) for ind, system in enumerate(systems)
        }

        self._couplings = {}
        self._coupled_systems = {}

        self.system_dims = dims
        self.dim = math.prod(dims)

    def __getitem__(self, label: str) -> System:
        return self._systems[label]

    def __contains__(self, label: str) -> bool:
        return label in self._systems

    def __len__(self) -> int:
        return len(self._systems)

    @property
    def systems(self) -> List[System]:
        """
        systems Returns the list of systems in the device.

        Returns
        -------
        List[System]
            The list of systems in the device.
        """
        return list(self._systems.values())

    @property
    def system_labels(self) -> List[str]:
        """
        system_labels Returns the labels of the systems in the device.

        Returns
        -------
        List[str]
            The labels of the systems in the device.
        """
        return list(self._systems.keys())

    @property
    def num_systems(self) -> int:
        """
        num_systems Returns the number of systems in the device.

        Returns
        -------
        int
            The number of systems in the device.
        """
        return len(self._systems)

    @property
    def couplings(self) -> List[Coupling]:
        """
        couplings Returns the list of couplings in the device.

        Returns
        -------
        List[Coupling]
            The list of couplings in the device.
        """
        return list(self._couplings.values())

    @property
    def coupling_labels(self) -> List[str]:
        """
        coupling_labels Returns the labels of the couplings in the device.

        Returns
        -------
        List[str]
            The labels of the couplings in the device.
        """
        return list(self._couplings.keys())

    @property
    def num_couplings(self) -> int:
        """
        num_couplings Returns the number of couplings in the device.

        Returns
        -------
        int
            The number of couplings in the device.
        """
        return len(self._couplings)

    def get_system(self, label: str) -> System:
        """
        get_system Returns a system from the device.

        Parameters
        ----------
        label : str
            The label of the system to be returned.

        Returns
        -------
        System
            The system with the given label.
        """
        return self._systems[label]

    def get_system_index(self, label: str) -> int | None:
        """
        get_system_index Returns the index of a system in the device.

        Parameters
        ----------
        label : str
            The label of the system.

        Returns
        -------
        int | None
            The index of the system.
        """
        return self._systems[label].device_ind

    def add_coupling(self, system: str, other_system: str, coupling: Coupling) -> None:
        """
        add_coupling Adds a coupling to the device.

        Parameters
        ----------
        coupling : Coupling
            The coupling to be added.
        """
        self._couplings[coupling.label] = coupling
        self._coupled_systems[coupling.label] = (system, other_system)

    def get_coupling(self, label: str) -> Coupling:
        """
        get_coupling Returns a coupling from the device.

        Parameters
        ----------
        label : str
            The label of the coupling to be returned.

        Returns
        -------
        Coupling
            The coupling with the given label.
        """
        return self._couplings[label]

    def get_bare_hamiltonian(self) -> Array:
        """
        get_bare_hamiltonian Returns the bare Hamiltonian of the device.
        This function will also transform the Hamiltonian into the diagonalized basis if the device is diagonalized or truncate the Hamiltonian if the device is truncated.

        Returns
        -------
        Array
            The bare Hamiltonian of the device.
        """
        hamiltonian_shape = (self.dim, self.dim)
        bare_hamiltonian = jnp.zeros(hamiltonian_shape)
        for system in self.systems:
            system_hamiltonian = system.get_hamiltonian()
            bare_hamiltonian = jnp.add(bare_hamiltonian, system_hamiltonian)
        return bare_hamiltonian

    def get_int_hamiltonian(self) -> Array:
        """
        get_int_hamiltonian Returns the interaction Hamiltonian of the device.

        Returns
        -------
        Array
            The interaction Hamiltonian of the device.
        """
        hamiltonian_shape = (self.dim, self.dim)
        int_hamiltonian = jnp.zeros(hamiltonian_shape)

        for coupling_label, coupling in self._couplings.items():
            label, other_label = self._coupled_systems[coupling_label]
            coupling_hamiltonian = coupling.get_hamiltonian(
                self._systems[label], self._systems[other_label]
            )
            int_hamiltonian += coupling_hamiltonian
        return int_hamiltonian

    def get_hamiltonian(self) -> Array:
        """
        get_himiltonian Returns the full Hamiltonian of the device.

        Returns
        -------
        Array
            The full Hamiltonian of the device.
        """
        bare_hamiltonian = self.get_bare_hamiltonian()
        int_hamiltonian = self.get_int_hamiltonian()
        return bare_hamiltonian + int_hamiltonian

    def get_drive_hamiltonian(self, time: ScalarLike) -> Array:
        """
        get_drive_hamiltonian Returns the drive Hamiltonian of the device.

        Parameters
        ----------
        time : ScalarLike
            The time at which the drive Hamiltonian is evaluated.

        Returns
        -------
        Array
            The drive Hamiltonian of the device.
        """
        hamiltonian_shape = (self.dim, self.dim)
        drive_hamiltonian = jnp.zeros(hamiltonian_shape)

        for system in self.systems:
            drive_hamiltonian += system.get_drive_hamiltonian(time)
        return drive_hamiltonian

    def get_eigenvalues(self) -> Array:
        """
        get_eigenvalues Returns the eigenvalues of the Hamiltonian of the quantum system.

        Returns
        -------
        Array
            The eigenvalues of the Hamiltonian.
        """
        hamiltonian = self.get_hamiltonian()
        eig_vals = eigh(hamiltonian, eigvals_only=True)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals

    def get_eigenstates(self) -> Tuple[Array, Array]:
        """
        get_eigenstates Returns the eigenvalues and eigenvectors
        of the Hamiltonian of the quantum system.

        Returns
        -------
        Tuple[Array, Array]
            The eigenvalues and eigenvectors of the Hamilton
        """
        hamiltonian = self.get_hamiltonian()
        eig_vals, eig_states = eigh(hamiltonian)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals, eig_states

    def get_bare_hamiltonian_qarray(self) -> ConstantTimeQArray:
        """
        get_bare_hamiltonian_qarray Returns the bare Hamiltonian as a TimeArray.

        Returns
        -------
        TimeArray
            The bare Hamiltonian as a TimeArray.
        """
        bare_hamiltonian = self.get_bare_hamiltonian()
        return constant(bare_hamiltonian)

    def get_hamiltonian_qarray(self) -> ConstantTimeQArray:
        """
        get_hamiltonian_qarray Returns the Hamiltonian as a TimeArray.

        Returns
        -------
        TimeArray
            The Hamiltonian as a TimeArray.
        """
        hamiltonian = self.get_hamiltonian()
        return constant(hamiltonian)

    def get_drive_qarray(self) -> SummedTimeQArray:
        """
        get_drive_hamiltonian_qarray Returns the drive Hamiltonian as a TimeArray.

        Returns
        -------
        TimeArray
            The drive Hamiltonian as a TimeArray.
        """
        time_arrays = [system.get_drive_qarray() for system in self.systems]
        return SummedTimeQArray(time_arrays)
