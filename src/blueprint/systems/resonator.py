from __future__ import annotations

import math
from typing import Tuple, Iterable

from scipy.constants import e, hbar

from jax import numpy as jnp
from jaxtyping import Array, Scalar

from .system import System
from ..operators import harmonic as harmonic_ops
from ..drives import Pulse, ChargeDrive, FluxDrive

type Float = float | Scalar


class Resonator(System):
    """Resonator model implementation."""

    _ec: Scalar
    _el: Scalar

    def __init__(
        self,
        label: str,
        charging_energy: Float,
        inductive_energy: Float,
        dim: int,
        device_ind: int | None = None,
        device_dims: Iterable[int] | None = None,
    ) -> None:
        super().__init__(label, dim, device_ind, device_dims)
        self._ec = jnp.asarray(charging_energy)
        self._el = jnp.asarray(inductive_energy)

    @property
    def charging_energy(self) -> Scalar:
        """
        charging_energy Charging energy parameter E_C for the resonator.

        Returns
        -------
        Scalar
            Charging energy value.
        """
        return self._ec

    @property
    def inductive_energy(self) -> Scalar:
        """
        inductive_energy Inductive energy parameter for the resonator.

        Returns
        -------
        Scalar
            Inductive energy value.
        """
        return self._el

    @property
    def plasma_frequency(self) -> Array:
        """
        plasma_frequency Plasma frequency of the resonator (sqrt(8 E_C E_L)).

        Returns
        -------
        Array
            Plasma frequency value.
        """
        freq = jnp.sqrt(8 * self._ec * self._el)
        return freq

    @property
    def charge_zpf(self) -> Scalar:
        """
        charge_zpf Charge zero-point fluctuations for the resonator.

        Returns
        -------
        Scalar
            Charge zero-point fluctuation.
        """
        return (self._el / (32 * self._ec)) ** 0.25

    @property
    def flux_zpf(self) -> Scalar:
        """
        flux_zpf Flux zero-point fluctuations for the resonator.

        Returns
        -------
        Scalar
            Flux zero-point fluctuation.
        """
        return (2 * self._ec / self._el) ** 0.25

    def embed(self, device_ind: int, device_dims: Iterable[int]) -> Resonator:
        """
        embed Embed resonator into a larger device Hilbert space.

        Parameters
        ----------
        device_ind : int
            Embedding index of this resonator.
        device_dims : tuple
            Subsystem dimensions for the device.
        """

        embedded_resonator = self.__class__(
            label=self.label,
            charging_energy=self.charging_energy,
            inductive_energy=self.inductive_energy,
            dim=self.dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )

        for label, drive in self._drives.items():
            embedded_resonator._drives[label] = drive

        return embedded_resonator

    def process_op(self, operator: Array) -> Array:
        """
        process_op Process an operator according to system configuration (embed/diag).

        Parameters
        ----------
        operator : Array
            Operator to process (native basis).

        Returns
        -------
        Array
            Operator in current system representation.
        """
        return self.embed_op(operator)

    def get_raise_op(self) -> Array:
        """
        get_raise_op Return raising operator in the system's current representation.

        Returns
        -------
        Array
            Raising operator in current basis.
        """
        raise_op = harmonic_ops.get_raise_op(self.dim)
        return self.process_op(raise_op)

    def get_low_op(self) -> Array:
        """
        get_low_op Return lowering operator in the system's current representation.

        Returns
        -------
        Array
            Lowering operator in current basis.
        """
        low_op = harmonic_ops.get_low_op(self.dim)
        return self.process_op(low_op)

    def get_number_op(self) -> Array:
        """
        get_number_op Return number operator in the current representation.

        Returns
        -------
        Array
            Number operator in current basis.
        """
        number_op = harmonic_ops.get_number_op(self.dim)
        return self.process_op(number_op)

    def get_identity_op(self) -> Array:
        """
        get_identity_op Return identity operator in current representation.

        Returns
        -------
        Array
            Identity operator in current basis.
        """
        id_op = jnp.identity(self.dim)
        return self.embed_op(id_op)

    def get_charge_op(self) -> Array:
        """
        get_charge_op Return charge operator in current representation.

        Returns
        -------
        Array
            Charge operator in current basis.
        """
        charge_op = harmonic_ops.get_charge_op(self.charge_zpf, self.dim)
        processed_op = self.process_op(charge_op)
        return processed_op

    def get_flux_op(self) -> Array:
        """
        get_flux_op Return flux operator in current representation.

        Returns
        -------
        Array
            Flux operator in current basis.
        """
        flux_op = harmonic_ops.get_flux_op(self.flux_zpf, self.dim)
        processed_op = self.process_op(flux_op)
        return processed_op

    def get_hamiltonian(self) -> Array:
        number_op = harmonic_ops.get_number_op(self.dim)
        hamiltonian = self.plasma_frequency * number_op
        return self.process_op(hamiltonian)

    def get_eigenvalues(self) -> Array:
        prefactors = jnp.arange(self.dim)
        eig_vals = self.plasma_frequency * prefactors
        return eig_vals

    def get_eigenstates(self) -> Tuple[Array, Array]:
        eig_states = jnp.identity(self.dim, dtype=complex)
        prefactors = jnp.arange(self.dim, dtype=float)
        eig_vals = self.plasma_frequency * prefactors
        return eig_vals, eig_states

    def add_charge_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_charge_drive Adds a charge drive to the resonator.

        Parameters
        ----------
        label : str
            The label of the drive.
        pulse : Drive
            The drive to add.

        Returns
        -------
        Self
            The resonator with the added drive.
        """
        drive = ChargeDrive(label=label, pulse=pulse)
        self._drives[label] = drive

    def add_flux_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_flux_drive Adds a flux drive to the resonator.

        Parameters
        ----------
        label : str
            The label of the drive.
        pulse : Drive
            The drive to add.

        Returns
        -------
        Self
            The resonator with the added drive.
        """
        drive = FluxDrive(label=label, pulse=pulse)
        self._drives[label] = drive

    @classmethod
    def from_frequency(
        cls,
        label: str,
        frequency: Float,
        impedance: Float,
        dim: int,
        *,
        device_ind: int | None = None,
        device_dims: Iterable[int] | None = None,
    ) -> Resonator:
        """
        from_frequency Returns a Resonator object from the frequency and impedence of the resonator.

        Parameters
        ----------
        label : str
            The label of the resonator.
        frequency : Float
            The frequency of the resonator.
        impedance : Float
            The characteristic impedance of the resonator.
        dim : int, optional
            The dimensionality of the resonator.


        Returns
        -------
        Resonator
            The resulting Resonator.
        """
        frequency = jnp.array(frequency)
        impedance = jnp.array(impedance)

        capacitance = 1 / (impedance * frequency)

        redifined_e = e / math.sqrt(hbar)
        charging_energy = (redifined_e**2) / (2 * capacitance)

        inductance = impedance / frequency
        inductive_energy = 1 / (4 * (redifined_e**2) * inductance)

        oscillator = cls(
            label, charging_energy, inductive_energy, dim, device_ind, device_dims
        )
        return oscillator
