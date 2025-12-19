from abc import abstractmethod
from typing import Callable

from jaxtyping import Array, Scalar
from equinox import Module, field

from dynamiqs.time_qarray import (
    constant,
    modulated,
    ConstantTimeQArray,
    ModulatedTimeQArray,
)
from ..systems import System

type Pulse = Callable[[float], Scalar | Array]


class Coupling(Module):
    """Base class for static coupling terms in device Hamiltonians."""

    label: str = field(static=True)
    strength: float | Scalar

    def __init__(self, label: str, strength: float | Scalar) -> None:
        self.label = label
        self.strength = strength

    @abstractmethod
    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Return coupling operator between two systems.

        Parameters
        ----------
        system : System
            First system in the coupling.
        other_system : System
            Second system in the coupling.

        Returns
        -------
        Array
            Coupling operator on the joint Hilbert space.
        """

    def get_hamiltonian(self, system: System, other_system: System) -> Array:
        """
        get_hamiltonian Return the (static) Hamiltonian term for the coupling.

        Returns
        -------
        Array
            Hamiltonian contribution from this coupling.
        """
        coupling_op = self.get_coupling_op(system, other_system)
        hamiltonian = self.strength * coupling_op
        return hamiltonian

    def get_hamiltonian_qarray(
        self, system: System, other_system: System
    ) -> ConstantTimeQArray:
        """
        get_hamiltonian_qarray Return the coupling Hamiltonian wrapped as a constant TimeQArray.

        Parameters
        ----------
        system : System
            First system in the coupling.
        other_system : System
            Second system in the coupling.

        Returns
        -------
        ConstantTimeQArray
            TimeQArray representation of the coupling Hamiltonian.
        """
        hamiltonian = self.get_hamiltonian(system, other_system)
        return constant(hamiltonian)


class TunableCoupling(Module):
    """Base class for time-dependent (tunable) coupling terms."""

    label: str = field(static=True)
    pulse: Pulse

    @abstractmethod
    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """get_coupling_op Return the (time-independent) coupling operator between systems."""

    def get_hamiltonian(
        self, time: float, system: System, other_system: System
    ) -> Array:
        """
        get_hamiltonian Return the time-dependent Hamiltonian term for the tunable coupling.

        Parameters
        ----------
        time : float
            Time at which the coupling strength is evaluated.
        system, other_system : System
            Coupled systems.

        Returns
        -------
        Array
            Time-dependent Hamiltonian contribution.
        """
        coupling_op = self.get_coupling_op(system, other_system)
        strength = self.pulse(time)
        hamiltonian = strength * coupling_op
        return hamiltonian

    def get_hamiltonian_qarray(
        self, system: System, other_system: System
    ) -> ModulatedTimeQArray:
        """
        get_hamiltonian_qarray Return modulated TimeQArray representing the tunable coupling.

        Parameters
        ----------
        system, other_system : System
            Coupled systems.

        Returns
        -------
        ModulatedTimeQArray
            TimeQArray object for the tunable coupling.
        """
        coupling_op = self.get_coupling_op(system, other_system)
        return modulated(self.pulse, coupling_op)
