from abc import abstractmethod
from typing import Callable, Tuple

from jax import numpy as jnp

from jaxtyping import Array, Scalar, ScalarLike

from equinox import Module, AbstractVar, field
from dynamiqs.time_qarray import (
    TimeQArray,
    modulated,
    SummedTimeQArray,
    ModulatedTimeQArray,
)

from ..systems import System

type Pulse = Callable[[float], Scalar | Array]


class BaseDrive(Module):
    label: AbstractVar[str]

    @abstractmethod
    def get_hamiltonian_qarray(self, system: System) -> TimeQArray:
        """
        get_hamiltonian_qarray Returns the drive Hamiltonian as a modulated dynamiqs.TimeArray object.

        Parameters
        ----------
        system : System
            The quantum system that the drive is acting on.

        Returns
        -------
        TimeArray
            The modulated dynamiqs.TimeArray object representing the drive Hamiltonian.
        """

    @abstractmethod
    def get_hamiltonian(self, system: System, time: ScalarLike) -> Array:
        """
        get_hamiltonian Returns the drive Hamiltonian evaluated at a given time.

        Parameters
        ----------
        system : System
            The quantum system that the drive is acting on.
        time : ScalarLike
            The time at which to evaluate the pulse amplitude.

        Returns
        -------
        Array
            The drive Hamiltonian evaluated at the given time.
        """


class Drive(BaseDrive):
    pulse: AbstractVar[Pulse]

    @abstractmethod
    def get_drive_op(self, system: System) -> Array:
        """
        get_drive_op Returns the operator that the drive is acting on.

        Parameters
        ----------
        system : System
            The quantum system that the drive is acting on.

        Returns
        -------
        Array
            The operator that the drive is acting on.
        """

    def get_hamiltonian_qarray(self, system) -> ModulatedTimeQArray:
        drive_op = self.get_drive_op(system)
        time_array = modulated(self.pulse, drive_op)
        return time_array

    def get_hamiltonian(self, system, time: ScalarLike) -> Array:
        drive_op = self.get_drive_op(system)
        pulse_val = self.pulse(time)
        return pulse_val * drive_op


class CompositeDrive(BaseDrive):
    """Abstract class for a composite drives that can be expressed as a sum of products of operators and their corresponding time-dependent prefactors."""

    label: str = field(static=True, converter=str)
    drives: Tuple[Drive, ...] = field(converter=tuple)

    def get_hamiltonian_qarray(self, system: System) -> SummedTimeQArray:
        time_arrays = [drive.get_hamiltonian_qarray(system) for drive in self.drives]
        time_array = SummedTimeQArray(time_arrays)
        return time_array

    def get_hamiltonian(self, system: System, time: ScalarLike) -> Array:
        hamiltonian_shape = (system.dim, system.dim)
        hamiltonian = jnp.zeros(hamiltonian_shape)

        for drive in self.drives:
            hamiltonian += drive.get_hamiltonian(system, time)

        return hamiltonian
