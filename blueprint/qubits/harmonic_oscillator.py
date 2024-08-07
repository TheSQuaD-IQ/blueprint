from __future__ import annotations
import math
from typing import Callable, Iterator

from scipy.constants import e, hbar
from jax import numpy as jnp
from jax import Array

from ..base import QuantumSystem


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
        decay_rate: float | None = None,
        deph_rate: float | None = None,
        thermal_photons: float = 0.0,
    ) -> None:
        """
        __init__ Initializes the HarmonicOscillator object.

        Parameters
        ----------
        label : str
            The label of the harmonic oscillator.
        charging_energy : float
            The charging energy of the oscillator.
        inductive_energy : float
            The inductive energy of the oscillator.
        dim : int, optional
            The dimension of the oscillator , by default 3
        decay_rate : float | None, optional
            The decay rate of the oscillator , by default None
        deph_rate : float | None, optional
            The pure dephasing rate of the oscillator, by default None
        thermal_photons : float | None, optional
            The thermal photon number of the environment leading to the decay of the oscillator, by default None

        Raises
        ------
        ValueError
            If the charging energy is not a float.
        ValueError
            If the charging energy is less than or equal to zero.
        ValueError
            If the inductive energy is not a float.
        ValueError
            If the inductive energy is less than or equal to zero.
        ValueError
            If the decay rate is not a float.
        ValueError
            If the decay rate is less than zero.
        ValueError
            If the dephasing rate is not a float.
        ValueError
            If the dephasing rate is less than zero.
        ValueError
            If the thermal photons is not a float.
        ValueError
            If the thermal photons is less than zero.
        """
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
        if decay_rate is not None:
            if not isinstance(decay_rate, float):
                raise ValueError(
                    f"The decay rate must be a float, instead got type {type(decay_rate)}."
                )
            if decay_rate < 0.0:
                raise ValueError("The decay rate must be greater than zero.")
        self._decay_rate = decay_rate

        if deph_rate is not None:
            if not isinstance(deph_rate, float):
                raise ValueError(
                    f"The dephasing rate must be a float, instead got type {type(deph_rate)}."
                )
            if deph_rate < 0.0:
                raise ValueError("The dephasing rate must be greater than zero.")
        self._deph_rate = deph_rate

        if not isinstance(thermal_photons, float):
            raise ValueError(
                f"The thermal photons must be a float, instead got type {type(thermal_photons)}."
            )
        if thermal_photons < 0.0:
            raise ValueError("The thermal photons must be greater than zero.")
        self._n_thermal = thermal_photons

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
                f"The inductive energy must be a float, instead got type {type(inductive_energy)}."
            )
        if inductive_energy <= 0.0:
            raise ValueError("The inductive energy must be greater than zero.")
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
        return (self._el / (32 * self._ec)) ** 0.25

    @property
    def flux_zpf(self) -> float:
        """
        flux_zpf Returns the zero-point fluctuations of the flux.

        Returns
        -------
        float
            The zero-point fluctuations of the flux.
        """
        return (2 * self._ec / self._el) ** 0.25

    def _get_raise_op(self) -> Array:
        """
        _get_creation_op Returns the raising (creation) operator of the transmon.

        Returns
        -------
        Array
            The raising (creation) operator of the transmon.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self._dim))
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
        offdiag = jnp.sqrt(jnp.arange(1, self._dim))
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

    def _get_number_op(self) -> Array:
        """
        _get_number_op Returns the number operator of the transmon.

        Returns
        -------
        Array
            The number operator of the transmon.
        """
        diagonal = jnp.arange(self._dim)
        return jnp.diag(diagonal)

    def get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the transmon.

        Returns
        -------
        Array
            The number operator in the current basis of the transmon.
        """
        number_op = self._get_number_op()
        return self.process_op(number_op)

    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Returns the Hamiltonian of the transmon in the charge basis.

        Returns
        -------
        Array
            The Hamiltonian of the transmon expressed in the charge basis.
        """
        number_op = self._get_number_op()
        hamiltonian = self.frequency * number_op
        return hamiltonian

    def _get_charge_op(self, include_fluctuations: bool) -> Array:
        """
        _get_charge_op Returns the charge operator of the transmon in the Fock basis.

        Returns
        -------
        Array
            The charge operator of the transmon, expressed in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        charge_op = 1.0j * (raise_op - low_op)
        if include_fluctuations:
            return self.charge_zpf * charge_op
        return charge_op

    def get_charge_op(self, *, include_fluctuations: bool = True) -> Array:
        """
        charge_op Returns the charge operator of the transmon.

        Returns
        -------
        Array
            The charge operator, in the current basis of the transmon.
        """
        charge_op = self._get_charge_op(include_fluctuations)
        return self.process_op(charge_op)

    def _get_flux_op(self, include_fluctuations: bool) -> Array:
        """
        _get_flux_op Returns the flux operator of the transmon in the Fock basis.

        Returns
        -------
        Array
            The flux operator of the transmon, expressed in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        flux_op = raise_op + low_op
        if include_fluctuations:
            return self.flux_zpf * flux_op
        return flux_op

    def get_flux_op(self, *, include_fluctuations: bool = True) -> Array:
        """
        get_flux_op Returns the flux operator of the transmon.

        Returns
        -------
        Array
            The flux operator, in the current basis of the transmon.
        """
        flux_op = self._get_flux_op(include_fluctuations)
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

    def add_charge_drive(
        self,
        label: str,
        charge_pulse: Callable,
        *,
        include_fluctuations: bool = True,
    ) -> None:
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

        charge_op = self._get_charge_op(include_fluctuations)
        self.add_drive(label, charge_pulse, charge_op)

    def _get_decay_ops(self) -> Iterator[Array]:
        """
        _get_decay_ops Yields the decay (and excitation) jump operators of the harmonic oscillator in the native fock bais.

        Yields
        ------
        Iterator[Array]
            The decay (and excitation) jump operators of the harmonic oscillator.

        Raises
        ------
        ValueError
            If the decay rate of the harmonic oscillator has not been set.
        """
        if self._decay_rate is None:
            raise ValueError("The decay rate of the transmon has not been set.")

        decay_prefactor = math.sqrt(self._decay_rate * (1 + self._n_thermal))
        low_op = self._get_low_op()

        decay_op = decay_prefactor * low_op
        yield decay_op

        if self._n_thermal > 0.0:
            exc_prefactor = math.sqrt(self._n_thermal * self._decay_rate)
            raise_op = self._get_raise_op()

            exc_op = exc_prefactor * raise_op
            yield exc_op

    def get_decay_ops(self) -> Iterator[Array]:
        """
        get_decay_ops Yield the decay (and excitation) jump operators of the harmonic oscillator, expressed in the transformed/truncated basis of the oscillator.

        Yields
        ------
        Iterator[Array]
            The decay (and excitation) jump operators of the harmonic oscillator.
        """
        decay_ops = self._get_decay_ops()
        for decay_op in decay_ops:
            yield self.process_op(decay_op)

    def _get_deph_ops(self) -> Iterator[Array]:
        """
        _get_deph_ops Yields the dephasing jump operators of the harmonic oscillator in the native fock basis.

        Yields
        ------
        Iterator[Array]
            The dephasing jump operators of the harmonic oscillator.

        Raises
        ------
        ValueError
            If the dephasing rate of the harmonic oscillator has not been set.
        """
        if self._deph_rate is None:
            raise ValueError("The deph rate of the transmon has not been set.")

        prefactor = math.sqrt(2 * self._deph_rate)
        number_op = self._get_number_op()

        deph_op = prefactor * number_op
        yield deph_op

    def get_deph_ops(self) -> Iterator[Array]:
        """
        get_deph_ops Yield the dephasing jump operators of the harmonic oscillator, expressed in the transformed/truncated basis of the oscillator.

        Yields
        ------
        Iterator[Array]
            The dephasing jump operators of the harmonic oscillator.
        """
        deph_ops = self._get_deph_ops()
        for deph_op in deph_ops:
            yield self.process_op(deph_op)

    def get_jump_ops(self) -> Iterator[Array]:
        """
        get_jump_ops Yields the jump operators associated with the Kerr non-linear oscillator.
        These correspond to either or both the energy relaxation and dephasing processes, depending on whether the values of the relaxation and dephasing times were provided, respectively.

        Yields
        ------
        Iterator[Array]
            The jump operators associated with the Kerr non-linear oscillator.
        """
        if self._decay_rate is not None:
            decay_ops = self.get_decay_ops()
            yield from decay_ops

        if self._deph_rate is not None:
            deph_ops = self.get_deph_ops()
            yield from deph_ops

    @staticmethod
    def from_frequency(
        label: str,
        frequency: float,
        impedance: float,
        dim: int = 3,
        decay_rate: float | None = None,
        deph_rate: float | None = None,
        thermal_photons: float = 0.0,
    ) -> HarmonicOscillator:
        """
        from_frequency Returns a HarmonicOscillator object from the frequency and impedence of the resonator.

        Parameters
        ----------
        label : str
            The label of the harmonic oscillator.
        frequency : float
            The frequency of the harmonic oscillator.
        impedance : float
            The characteristic impedance of the harmonic oscillator.
        dim : int, optional
            The dimensionality of the harmonic oscillator , by default 3
        decay_rate : float | None, optional
            The decay rate of the harmonic resonator, by default None
        deph_rate : float | None, optional
            The pure dephasing rate of the harmonic oscillatgor, by default None


        Returns
        -------
        HarmonicOscillator
            The resulting HarmonicOscillator .

        Raises
        ------
        ValueError
            If the frequency is not a float.
        ValueError
            If the frequency is less than or equal to zero.
        ValueError
            If the impedence is not a float.
        ValueError
            If the impedence is less than or equal to zero.
        """
        if not isinstance(frequency, float):
            raise ValueError(
                f"The frequency must be a float, instead got type {type(frequency)}."
            )
        if frequency <= 0.0:
            raise ValueError("The frequency must be greater than zero.")

        if not isinstance(impedance, float):
            raise ValueError(
                f"The impedence must be a float, instead got type {type(impedance)}."
            )
        if impedance <= 0.0:
            raise ValueError("The impedence must be greater than zero.")
        capacitance = 1 / (impedance * frequency)

        redifined_e = e / math.sqrt(hbar)
        charging_energy = (redifined_e**2) / (2 * capacitance)

        inductance = impedance / frequency
        inductive_energy = 1 / (4 * (redifined_e**2) * inductance)

        oscillator = HarmonicOscillator(
            label=label,
            charging_energy=charging_energy,
            inductive_energy=inductive_energy,
            dim=dim,
            decay_rate=decay_rate,
            deph_rate=deph_rate,
            thermal_photons=thermal_photons,
        )
        return oscillator
