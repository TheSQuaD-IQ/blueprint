from __future__ import annotations
import math
from typing import Callable

from jax import numpy as jnp
from jax import Array

from ..base import QuantumSystem
from ..drives import Drive


class HarmonicOscillator(QuantumSystem):
    """
    HarmonicOscillator An approximate trasmon model as an LC HarmonicOscillator.

    """

    def __init__(
        self,
        label: str,
        charging_energy: float,
        inductive_energy: float,
        dim: int = 3,
        relax_time: float | None = None,
        deph_time: float | None = None,
    ) -> None:
        if not isinstance(charging_energy, float):
            raise ValueError(
                f"The charging energy must be a float, instead got type {type(charging_energy)}."
            )
        if charging_energy <= 0.0:
            raise ValueError("The charging energy must be greater than zero.")
        self._ec = charging_energy

        if not isinstance(inductive_energy, float):
            raise ValueError(
                f"The inductive energy must be a float, instead got type {type(inductive_energy)}."
            )
        if inductive_energy <= 0.0:
            raise ValueError("The inductive energy must be greater than zero.")
        self._el = inductive_energy
        super().__init__(label, dim)

        # The relaxation and dephasing times
        if relax_time is not None:
            if relax_time <= 0.0:
                raise ValueError("The relaxation time must be greater than zero.")
        self._relax_time = relax_time

        if deph_time is not None:
            if deph_time <= 0.0:
                raise ValueError("The dephasing time must be greater than zero.")
            if relax_time is not None:
                if deph_time > 2 * relax_time:
                    raise ValueError(
                        "The dephasing time must be less than or equal to two times the relaxation time."
                    )
        self._deph_time = deph_time

    @property
    def charging_energy(self) -> float:
        """
        charging_energy Returns the charging energy of the oscillator.

        Returns
        -------
        float
            The charging energy of the oscillator.
        """
        return self._ec

    @charging_energy.setter
    def charging_energy(self, charging_energy: float) -> None:
        """
        charging_energy Sets the charging energy of the oscillator.

        Parameters
        ----------
        charging_energy : float
            The charging energy of the oscillator.

        Raises
        ------
        ValueError
            If the charging energy is not a float.
        ValueError
            If the charging energy is less than or equal to zero.
        """
        if not isinstance(charging_energy, float):
            raise ValueError(
                f"The charging energy must be a float, instead got type {type(charging_energy)}."
            )
        if charging_energy <= 0.0:
            raise ValueError("The charging energy must be greater than zero.")
        self._ec = charging_energy

    @property
    def inductive_energy(self) -> float:
        """
        inductive_energy Returns the inductive energy of the oscillator.

        Returns
        -------
        float
            The inductive energy of the oscillator.
        """
        return self._el

    @inductive_energy.setter
    def inductive_energy(self, inductive_energy: float) -> None:
        """
        inductive_energy Sets the inductive energy of the oscillator.

        Parameters
        ----------
        inductive_energy : float
            The inductive energy of the oscillator.

        Raises
        ------
        ValueError
            If the inductive energy is not a float.
        ValueError
            If the inductive energy is less than or equal to zero.
        """
        if not isinstance(inductive_energy, float):
            raise ValueError(
                f"The Josephson energy must be a float, instead got type {type(inductive_energy)}."
            )
        if inductive_energy <= 0.0:
            raise ValueError("The Josephson energy must be greater than zero.")
        self._el = inductive_energy

    @property
    def frequency(self) -> float:
        """
        max_frequency Returns the maximum frequency of the transmon.

        Returns
        -------
        float
            The maximum frequency of the transmon.
        """
        freq = math.sqrt(8 * self._ec * self._el)
        return freq

    @property
    def charge_zpf(self) -> float:
        """
        charge_zpf Returns the zero-point fluctuations of the charge.

        Returns
        -------
        float
            The zero-point fluctuations of the charge.
        """
        return (self.inductive_energy / (32 * self.charging_energy)) ** 0.25

    @property
    def flux_zpf(self) -> float:
        """
        flux_zpf Returns the zero-point fluctuations of the flux.

        Returns
        -------
        float
            The zero-point fluctuations of the flux.
        """
        return (2 * self.charging_energy / self.inductive_energy) ** 0.25

    def _get_raise_op(self) -> Array:
        """
        _get_creation_op Returns the raising (creation) operator of the transmon.

        Returns
        -------
        Array
            The raising (creation) operator of the transmon.
        """
        dim = self._trunc_dim or self._dim
        offdiag = jnp.sqrt(jnp.arange(1, dim))
        return jnp.diag(offdiag, k=-1)

    def get_raise_op(self) -> Array:
        """
        get_creation_op Returns the raising (creation) operator of the transmon.

        Returns
        -------
        Array
            The raising (creation) operator in the current basis of the transmon.
        """
        raise_op = self._get_raise_op()
        return self.process_op(raise_op)

    def _get_low_op(self) -> Array:
        """
        _get_low_op Returns the lowering (annihilaton) operator of the transmon.

        Returns
        -------
        Array
            The lowering (annihilaton) operator of the transmon.
        """
        dim = self._trunc_dim or self._dim
        offdiag = jnp.sqrt(jnp.arange(1, dim))
        return jnp.diag(offdiag, k=1)

    def get_low_op(self) -> Array:
        """
        get_low_op Returns the creation operator of the transmon.

        Returns
        -------
        Array
            The lowering (annihilaton) operator in the current basis of the transmon.
        """
        low_op = self._get_low_op()
        return self.process_op(low_op)

    def _get_num_op(self) -> Array:
        """
        _get_num_op Returns the number operator of the transmon.

        Returns
        -------
        Array
            The number operator of the transmon.
        """
        dim = self._trunc_dim or self._dim
        diagonal = jnp.arange(dim)
        return jnp.diag(diagonal)

    def get_num_op(self) -> Array:
        """
        get_num_op Returns the number operator of the transmon.

        Returns
        -------
        Array
            The number operator in the current basis of the transmon.
        """
        num_op = self._get_num_op()
        return self.process_op(num_op)

    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Returns the Hamiltonian of the transmon in the charge basis.

        Returns
        -------
        Array
            The Hamiltonian of the transmon expressed in the charge basis.
        """
        dim = self._trunc_dim or self._dim
        num_op = self._get_num_op()
        id_op = jnp.identity(dim)

        hamiltonian = self.frequency * (num_op + 0.5 * id_op)
        return hamiltonian

    def _get_charge_op(self) -> Array:
        """
        _get_charge_op Returns the charge operator of the transmon in the Fock basis.

        Returns
        -------
        Array
            The charge operator of the transmon, expressed in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        charge_op = 1.0j * self.charge_zpf * (raise_op - low_op)
        return charge_op

    def get_charge_op(self) -> Array:
        """
        charge_op Returns the charge operator of the transmon.

        Returns
        -------
        Array
            The charge operator, in the current basis of the transmon.
        """
        charge_op = self._get_charge_op()
        return self.process_op(charge_op)

    def _get_flux_op(self) -> Array:
        """
        _get_flux_op Returns the flux operator of the transmon in the Fock basis.

        Returns
        -------
        Array
            The flux operator of the transmon, expressed in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        return self.flux_zpf * (raise_op + low_op)

    def get_flux_op(self) -> Array:
        """
        get_flux_op Returns the flux operator of the transmon.

        Returns
        -------
        Array
            The flux operator, in the current basis of the transmon.
        """
        flux_op = self._get_flux_op()
        return self.process_op(flux_op)

    def get_potential(self, flux: float | Array) -> float | Array:
        """
        potential Returns the potential energy of the transmon.

        Parameters
        ----------
        flux : float | Array
            The flux values.

        Returns
        -------
        float | Array
            The potential energy.
        """
        potential = 0.5 * self._el * flux**2
        return potential

    def add_charge_drive(self, label: str, charge_pulse: Callable) -> None:
        """
        add_charge_drive Applies a charge drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        charge_pulse : Callable
            The time-dependent charge pulse applied to the transmon. This must be a
            callable object that returns the applied charge pulse as a function of time.

        Raises
        ------
        ValueError
            If a drive with the same label has already been
        """
        if label in self._drives:
            raise ValueError(
                f"A drive with the label '{label}' has already been applied to the transmon."
            )

        if not isinstance(charge_pulse, Callable):
            raise ValueError(
                f"The charge pulse must be either a float or a Callable object, instead got type {type(charge_pulse)}."
            )

        charge_op = self._get_charge_op()

        drive = Drive(label, charge_pulse, charge_op)
        self._drives[label] = drive

    @staticmethod
    def from_frequency(
        label: str,
        frequency: float,
        dim: int = 3,
        relax_time: float | None = None,
        deph_time: float | None = None,
    ) -> HarmonicOscillator:
        """
        from_energies Create a HarmonicOscillator from the charging and Josephson energies. This is so far assuming symmetric junctions.

        Parameters
        ----------
        label : str
            The label of the transmon.
        charge_energy : float
            The charging energy of the transmon.
        joseph_energy : float
            The Josephson energy of the transmon.
        ext_flux : float, optional
            The external flux through the SQUID loop of the transmon, by default 0.0
        asymmetry : float, optional
            The asymmetry of the junction, by default 0.0
        dim : int, optional
            The dimension of the Hilbert space of the transmon, by default 3
        relax_time : float, optional
            The relaxation time of the transmon, by default None
        deph_time : float, optional
            The dephasing time of the transmon, by default None



        Returns
        -------
        HarmonicOscillator
            The HarmonicOscillator instance.
        """
        raise NotImplementedError("This method is not implemented yet.")
