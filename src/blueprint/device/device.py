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
        systems List systems in the device.

        Returns
        -------
        List[System]
            Systems contained in the device.
        """
        return list(self._systems.values())

    @property
    def system_labels(self) -> List[str]:
        """
        system_labels Labels of systems in the device.

        Returns
        -------
        List[str]
            Labels of systems in the device.
        """
        return list(self._systems.keys())

    @property
    def num_systems(self) -> int:
        """
        num_systems Number of systems in the device.

        Returns
        -------
        int
            Count of systems.
        """
        return len(self._systems)

    @property
    def couplings(self) -> List[Coupling]:
        """
        couplings List couplings registered with the device.

        Returns
        -------
        List[Coupling]
            Coupling instances.
        """
        return list(self._couplings.values())

    @property
    def coupling_labels(self) -> List[str]:
        """
        coupling_labelsLabels of couplings in the device.

        Returns
        -------
        List[str]
            Coupling labels.
        """
        return list(self._couplings.keys())

    @property
    def num_couplings(self) -> int:
        """
        num_couplings Number of couplings in the device.

        Returns
        -------
        int
            Count of couplings.
        """
        return len(self._couplings)

    def get_system(self, label: str) -> System:
        """
        get_system Return a `System` by label.

        Parameters
        ----------
        label : str
            Label of the requested system.

        Returns
        -------
        System
            System instance with matching label.
        """
        return self._systems[label]

    def get_system_index(self, label: str) -> int | None:
        """
        get_system_index Return device index for a named system.

        Parameters
        ----------
        label : str
            System label.

        Returns
        -------
        int or None
            Index of system within device embedding (or ``None``).
        """
        return self._systems[label].device_ind

    def add_coupling(self, system: str, other_system: str, coupling: Coupling) -> None:
        """
        add_coupling Add a coupling to the device.

        Parameters
        ----------
        system : str
            Label of the first system in the coupling.
        other_system : str
            Label of the second system in the coupling.
        coupling : Coupling
            Coupling instance to register.
        """
        self._couplings[coupling.label] = coupling
        self._coupled_systems[coupling.label] = (system, other_system)

    def get_coupling(self, label: str) -> Coupling:
        """
        get_coupling Return a registered `Coupling` by label.

        Parameters
        ----------
        label : str
            Label of the coupling.

        Returns
        -------
        Coupling
            Coupling instance.
        """
        return self._couplings[label]

    def get_bare_hamiltonian(self) -> Array:
        """
        get_bare_hamiltonian Compute the bare (non-interacting) Hamiltonian of the device.

        Returns
        -------
        Array
            Bare Hamiltonian matrix of the device.
        """
        hamiltonian_shape = (self.dim, self.dim)
        bare_hamiltonian = jnp.zeros(hamiltonian_shape)
        for system in self.systems:
            system_hamiltonian = system.get_hamiltonian()
            bare_hamiltonian = jnp.add(bare_hamiltonian, system_hamiltonian)
        return bare_hamiltonian

    def get_int_hamiltonian(self) -> Array:
        """
        get_int_hamiltonian Compute the interaction Hamiltonian from registered couplings.

        Returns
        -------
        Array
            Interaction Hamiltonian matrix.
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
        get_hamiltonian Return the total device Hamiltonian (bare + interaction).

        Returns
        -------
        Array
            Total Hamiltonian matrix for the device.
        """
        bare_hamiltonian = self.get_bare_hamiltonian()
        int_hamiltonian = self.get_int_hamiltonian()
        return bare_hamiltonian + int_hamiltonian

    def get_drive_hamiltonian(self, time: ScalarLike) -> Array:
        """
        get_drive_hamiltonian Return the total time-dependent drive Hamiltonian at `time`.

        Parameters
        ----------
        time : ScalarLike
            Time at which to evaluate drive Hamiltonians.

        Returns
        -------
        Array
            Drive Hamiltonian matrix at the given time.
        """
        hamiltonian_shape = (self.dim, self.dim)
        drive_hamiltonian = jnp.zeros(hamiltonian_shape)

        for system in self.systems:
            drive_hamiltonian += system.get_drive_hamiltonian(time)
        return drive_hamiltonian

    def get_eigenvalues(self) -> Array:
        """
        get_eigenvalues Return eigenvalues of the total device Hamiltonian, offset by the ground state.

        Returns
        -------
        Array
            Eigenvalues normalized to have ground state energy zero.
        """
        hamiltonian = self.get_hamiltonian()
        eig_vals = eigh(hamiltonian, eigvals_only=True)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals

    def get_eigenstates(self) -> Tuple[Array, Array]:
        """
        get_eigenstates Return eigenvalues and eigenvectors of the total Hamiltonian.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of (eigenvalues, eigenvectors).
        """
        hamiltonian = self.get_hamiltonian()
        eig_vals, eig_states = eigh(hamiltonian)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals, eig_states

    def get_bare_hamiltonian_qarray(self) -> ConstantTimeQArray:
        """
        get_bare_hamiltonian_qarray Return bare Hamiltonian wrapped in a constant TimeQArray.

        Returns
        -------
        ConstantTimeQArray
            Constant time-dependent Hamiltonian object.
        """
        bare_hamiltonian = self.get_bare_hamiltonian()
        return constant(bare_hamiltonian)

    def get_hamiltonian_qarray(self) -> ConstantTimeQArray:
        """
        get_hamiltonian_qarray Return total Hamiltonian wrapped in a constant TimeQArray.

        Returns
        -------
        ConstantTimeQArray
            Constant time-dependent Hamiltonian object.
        """
        hamiltonian = self.get_hamiltonian()
        return constant(hamiltonian)

    def get_drive_qarray(self) -> SummedTimeQArray:
        """
        get_drive_qarray Return summed TimeQArray of drive Hamiltonians from subsystems.

        Returns
        -------
        SummedTimeQArray
            TimeQArray representing total drive Hamiltonian.
        """
        time_arrays = [system.get_drive_qarray() for system in self.systems]
        return SummedTimeQArray(time_arrays)
