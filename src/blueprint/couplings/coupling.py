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
    """
    Coupling A class for representing a coupling in the Hamiltonian.
    """

    label: str = field(static=True)
    strength: float | Scalar

    def __init__(self, label: str, strength: float | Scalar) -> None:
        self.label = label
        self.strength = strength

    @abstractmethod
    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Returns the coupling operator between two systems.

        Parameters
        ----------
        system : System
            The first system in the coupling.
        other_system : System
            The second system in the coupling.

        Returns
        -------
        Array
            The coupling operator between the two systems.
        """

    def get_hamiltonian(self, system: System, other_system: System) -> Array:
        """
        get_hamiltonian Returns the Hamiltonian term corresponding to the coupling.

        Returns
        -------
        Array
            The Hamiltonian term corresponding to the coupling.
        """
        coupling_op = self.get_coupling_op(system, other_system)
        hamiltonian = self.strength * coupling_op
        return hamiltonian

    def as_qarray(self, system: System, other_system: System) -> ConstantTimeQArray:
        """
        as_qarray Returns the coupling Hamiltonian as a TimeArray object.

        Parameters
        ----------
        system : System
            The first system in the coupling.
        other_system : System
            The second system in the coupling.

        Returns
        -------
        TimeArray
            The TimeArray object representing the coupling Hamiltonian.
        """
        hamiltonian = self.get_hamiltonian(system, other_system)
        return constant(hamiltonian)


class TunableCoupling(Module):
    """
    Coupling A class for representing a coupling in the Hamiltonian.
    """

    label: str = field(static=True)
    pulse: Pulse

    @abstractmethod
    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Returns the coupling operator between two systems.

        Parameters
        ----------
        system : System
            The first system in the coupling.
        other_system : System
            The second system in the coupling.

        Returns
        -------
        Array
            The coupling operator between the two systems.
        """

    def get_hamiltonian(
        self, time: float, system: System, other_system: System
    ) -> Array:
        """
        get_hamiltonian Returns the Hamiltonian term corresponding to the coupling.

        Returns
        -------
        Array
            The Hamiltonian term corresponding to the coupling.
        """
        coupling_op = self.get_coupling_op(system, other_system)
        strength = self.pulse(time)
        hamiltonian = strength * coupling_op
        return hamiltonian

    def as_qarray(self, system: System, other_system: System) -> ModulatedTimeQArray:
        """
        as_qarray Returns the coupling Hamiltonian as a TimeArray object.

        Parameters
        ----------
        system : System
            The first system in the coupling.
        other_system : System
            The second system in the coupling.

        Returns
        -------
        TimeArray
            The TimeArray object representing the coupling Hamiltonian.
        """
        coupling_op = self.get_coupling_op(system, other_system)
        return modulated(self.pulse, coupling_op)
