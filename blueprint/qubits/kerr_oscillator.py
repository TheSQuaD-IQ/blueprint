from __future__ import annotations
import math
from typing import Union, Callable, Iterator
from functools import wraps

from jax import numpy as jnp
from jax import Array

from ..base import QuantumSystem


class KerrOscillator(QuantumSystem):
    """
    KerrOscillator An approximate trasmon model as an anharmonic oscillator.

    """

    def __init__(
        self,
        label: str,
        frequency: float,
        anharmonicity: float,
        ext_flux: float = 0.0,
        asymmetry: float = 0.0,
        dim: int = 3,
        relax_time: float | None = None,
        deph_time: float | None = None,
    ) -> None:
        if not isinstance(frequency, float):
            raise ValueError(
                f"The frequency must be a float, instead got type {type(frequency)}."
            )
        if frequency <= 0.0:
            raise ValueError("The frequency must be greater than zero.")
        self._freq = frequency

        if not isinstance(anharmonicity, float):
            raise ValueError(
                f"The anharmonicity must be a float, instead got type {type(anharmonicity)}."
            )
        if anharmonicity >= 0.0:
            raise ValueError("The anharmonicity must be smaller than 0.")
        self._anharm = anharmonicity

        if not isinstance(ext_flux, float):
            raise ValueError(
                f"The external flux must be a float, instead got type {type(ext_flux)}."
            )
        self._ext_flux = ext_flux

        if not isinstance(asymmetry, float):
            raise ValueError(
                f"The asymmetry must be a float, instead got type {type(asymmetry)}."
            )
        if not 0.0 <= asymmetry <= 1.0:
            raise ValueError("The asymmetry must be between 0 and 1.")
        self._asymm = asymmetry

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
    def approx_frequency(self) -> float:
        """
        frequency Returns the frequency of the transmon.

        Returns
        -------
        float
            The frequency of the transmon.
        """
        josephson_energy = self.eff_josephson_energy
        res_freq = math.sqrt(8 * self.charging_energy * josephson_energy)
        return res_freq - self.charging_energy

    @property
    def max_frequency(self) -> float:
        """
        max_frequency Returns the maximum frequency of the transmon.

        Returns
        -------
        float
            The maximum frequency of the transmon.
        """
        return self._freq

    @max_frequency.setter
    def max_frequency(self, frequency: float) -> None:
        """
        max_frequency Sets the maximum frequency of the transmon.

        Parameters
        ----------
        frequency : float
            The maximum frequency of the transmon.

        Raises
        ------
        ValueError
            If the frequency is not a float.
        ValueError
            If the frequency is less than or equal to zero.
        """
        if not isinstance(frequency, float):
            raise ValueError(
                f"The frequency must be a float, instead got type {type(frequency)}."
            )
        if frequency <= 0.0:
            raise ValueError("The frequency must be greater than zero.")
        self._freq = frequency

    @property
    def anharmonicity(self) -> float:
        """
        anharmonicity Returns the anharmonicity of the transmon.

        Returns
        -------
        float
            The anharmonicity of the transmon.
        """
        return self._anharm

    @anharmonicity.setter
    def anharmonicity(self, anharmonicity: float) -> None:
        """
        anharmonicity Sets the anharmonicity of the transmon.

        Parameters
        ----------
        anharmonicity : float
            The anharmonicity of the transmon.

        Raises
        ------
        ValueError
            If the anharmonicity is not a float.
        ValueError
            If the anharmonicity is greater than or equal to zero.
        """
        if not isinstance(anharmonicity, float):
            raise ValueError(
                f"The anharmonicity must be a float, instead got type {type(anharmonicity)}."
            )
        if anharmonicity >= 0.0:
            raise ValueError("The anharmonicity must be smaller than 0.")
        self._anharm = anharmonicity

    @property
    def ext_flux(self) -> float:
        """
        ext_flux Returns the external flux through the SQUID loop of the transmon.

        Returns
        -------
        float
            The external flux.
        """
        return self._ext_flux

    @ext_flux.setter
    def ext_flux(self, ext_flux: float) -> None:
        """
        ext_flux Sets the external flux through the SQUID loop of the transmon.

        Parameters
        ----------
        ext_flux : float
            The external flux.

        Raises
        ------
        ValueError
            If the external flux is not a float.
        """
        if not isinstance(ext_flux, float):
            raise ValueError(
                f"The external flux must be a float, instead got type {type(ext_flux)}."
            )
        self._ext_flux = ext_flux

    @property
    def asymmetry(self) -> float:
        """
        asymmetry Returns the asymmetry of the transmon.

        Returns
        -------
        float
            The asymmetry of the transmon.
        """
        return self._asymm

    @asymmetry.setter
    def asymmetry(self, asymmetry: float) -> None:
        """
        asymmetry Sets the asymmetry of the transmon.

        Parameters
        ----------
        asymmetry : float
            The asymmetry of the transmon.

        Raises
        ------
        ValueError
            If the asymmetry is not a float.
        ValueError
            If the asymmetry is not between 0 and 1.
        """
        if not isinstance(asymmetry, float):
            raise ValueError(
                f"The asymmetry must be a float, instead got type {type(asymmetry)}."
            )
        if not 0.0 <= asymmetry <= 1.0:
            raise ValueError("The asymmetry must be between 0 and 1.")
        self._asymm = asymmetry

    @property
    def charging_energy(self) -> float:
        """
        charging_energy Returns the charging energy of the transmon.

        Returns
        -------
        float
            The charging energy of the transmon.
        """
        return -self._anharm

    @property
    def josephson_energy(self) -> float:
        """
        josephson_energy Returns the Josephson energy of the transmon.

        Returns
        -------
        float
            The Josephson energy of the transmon.
        """
        ec = self.charging_energy
        joseph_energy = (self._freq + ec) ** 2 / (8 * ec)
        return joseph_energy

    def get_josephson_energy(self, ext_flux: float) -> float:
        """
        eff_josephson_energy Returns the effective Josephson energy of the transmon.
        This is the Josephson energy modified by the external flux and the junction asymmetry.

        Returns
        -------
        float
            The effective Josephson energy
        """
        cos_term = math.cos(ext_flux)
        sqrt_term = math.sqrt(1 + self._asymm**2 * math.tan(ext_flux) ** 2)

        prefactor = abs(cos_term) * sqrt_term
        return self.josephson_energy * prefactor

    @property
    def eff_josephson_energy(self) -> float:
        """
        eff_josephson_energy Returns the effective Josephson energy of the transmon.
        This is the Josephson energy modified by the external flux and the junction asymmetry.

        Returns
        -------
        float
            The effective Josephson energy.
        """
        return self.get_josephson_energy(self._ext_flux)

    @property
    def charge_zpf(self) -> float:
        """
        charge_zpf Returns the zero-point fluctuations of the charge.

        Returns
        -------
        float
            The zero-point fluctuations of the charge.
        """
        return (self.eff_josephson_energy / (32 * self.charging_energy)) ** 0.25

    @property
    def flux_zpf(self) -> float:
        """
        flux_zpf Returns the zero-point fluctuations of the flux.

        Returns
        -------
        float
            The zero-point fluctuations of the flux.
        """
        return (2 * self.charging_energy / self.eff_josephson_energy) ** 0.25

    def get_frequency(self, ext_flux: float) -> float:
        """
        get_frequency Returns the frequency shift of the transmon due to the applied flux.

        Parameters
        ----------
        flux : float
            The applied flux.

        Returns
        -------
        float
            The frequency shift.
        """
        cos_term = abs(math.cos(ext_flux))
        sqrt_term = math.sqrt(1 + self.asymmetry**2 * math.tan(ext_flux) ** 2)

        eff_ej = self.josephson_energy * cos_term * sqrt_term
        res_freq = math.sqrt(8 * self.charging_energy * eff_ej)

        shifted_freq = res_freq - self.charging_energy
        return shifted_freq

    @property
    def frequency(self) -> float:
        """
        approximate_frequency Returns the approximate 0-1 frequency of the transmon.

        Returns
        -------
        float
            The approximate transmon 0-1 frequency.
        """
        return self.get_frequency(self._ext_flux)

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

    def _get_number_op(self) -> Array:
        """
        _get_number_op Returns the number operator of the transmon.

        Returns
        -------
        Array
            The number operator of the transmon.
        """
        dim = self._trunc_dim or self._dim
        diagonal = jnp.arange(dim)
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

        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        number_op = self._get_number_op()

        oscillator_term = self.frequency * number_op

        anharm_op = raise_op @ raise_op @ low_op @ low_op
        anharmonic_term = 0.5 * self._anharm * anharm_op

        hamiltonian = oscillator_term + anharmonic_term
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

    def get_potential(self, phases: Union[float, Array]) -> Array:
        """
        potential Returns the potential energy of the transmon.

        Parameters
        ----------
        phases : Array
            The phase values.

        Returns
        -------
        Array
            The potential energy.
        """
        potential = -self.josephson_energy * jnp.cos(phases)
        return potential

    def add_flux_drive(self, label: str, flux_pulse: Callable, **keywords) -> None:
        """
        add_flux_drive Applies a flux drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        flux_pulse : Callable
            The time-dependent flux pulse applied to the transmon.
            This must be a callable object that returns the
            applied flux pulse as a function of time.

        Raises
        ------
        ValueError
            If a drive with the same label has already been
        """
        if not isinstance(flux_pulse, Callable):
            raise ValueError(
                f"The flux pulse must be either a float or a Callable object, instead got type {type(flux_pulse)}."
            )

        @wraps(flux_pulse)
        def prefactor(*args, **kwargs) -> Array:
            applied_flux = flux_pulse(*args, **kwargs)
            total_flux = self._ext_flux + applied_flux

            cos_term = jnp.abs(jnp.cos(total_flux))
            sqrt_term = jnp.sqrt(1 + self.asymmetry**2 * jnp.tan(total_flux) ** 2)

            eff_ej = self.josephson_energy * cos_term * sqrt_term
            res_freq = jnp.sqrt(8 * self.charging_energy * eff_ej)

            shifted_freq = res_freq - self.charging_energy
            freq_shift = shifted_freq - self.frequency
            return freq_shift

        number_op = self._get_number_op()
        self.add_drive(label, prefactor, number_op, **keywords)

    def add_charge_drive(
        self,
        label: str,
        charge_pulse: Callable,
        *,
        include_fluctuations: bool = True,
        **keywords,
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
        """
        charge_op = self._get_charge_op(include_fluctuations)
        self.add_drive(label, charge_pulse, charge_op, **keywords)

    def add_detuning_drive(
        self, label: str, detuning_pulse: Callable, **keywords
    ) -> None:
        """
        add_detuning_drive Applies a direct frequency detuning drive to the transmon.
        Note that this isn't a phyiscal drive, but rather a drive that directly detunes the qubit frequency.
        Typically, this would instead be realized by applying a flux pulse.
        However, the availaility of this method allows for additional utility and can be used to find
        optimal frequency detunings, which can then be converted to an applied flux instead.

        Parameters
        ----------
        label : str
            The label of the drive.
        detuning_pulse : Callable
            The time-dependent detuning pulse applied to the transmon. This must be a
            callable object that returns the applied frequency detuning pulse as a function of time.
        """
        number_op = self._get_number_op()
        self.add_drive(label, detuning_pulse, number_op, **keywords)

    def get_relaxation_op(self) -> Array:
        """
        get_relaxation_op Returns the relaxation jump operator of the transmon.

        Returns
        -------
        Array
            The relaxation jump operator.

        Raises
        ------
        ValueError
            If the relaxation time has not been set.
        """
        if self._relax_time is None:
            raise ValueError("The relaxation time has not been set.")
        relax_rate = 1 / self._relax_time
        prefactor = math.sqrt(relax_rate)

        low_op = self.get_low_op()
        relax_op = prefactor * low_op
        return self.process_op(relax_op)

    def get_dephasing_op(self) -> Array:
        if self._deph_time is None:
            raise ValueError("The dephasing time has not been set.")
        deph_rate = 1 / self._deph_time
        if self._relax_time is None:
            prefactor = math.sqrt(deph_rate)

            number_op = self._get_number_op()
            deph_op = prefactor * number_op
            return self.process_op(deph_op)

        relax_rate = 1 / self._relax_time
        pure_deph_rate = deph_rate - 0.5 * relax_rate
        prefactor = math.sqrt(pure_deph_rate)

        number_op = self._get_number_op()
        deph_op = prefactor * number_op
        return self.process_op(deph_op)

    def get_jump_ops(self) -> Iterator[Array]:
        """
        get_jump_ops Yields the jump operators associated with the Kerr non-linear oscillator.
        These correspond to either or both the energy relaxation and dephasing processes, depending on whether the values of the relaxation and dephasing times were provided, respectively.

        Yields
        ------
        Iterator[Array]
            The jump operators associated with the Kerr non-linear oscillator.
        """
        if self._relax_time is not None:
            relax_op = self.get_relaxation_op()
            yield relax_op

        if self._deph_time is not None:
            deph_op = self.get_dephasing_op()
            yield deph_op

    @staticmethod
    def from_energies(
        label: str,
        charge_energy: float,
        joseph_energy: float,
        ext_flux: float = 0.0,
        asymmetry: float = 0.0,
        dim: int = 3,
        relax_time: float | None = None,
        deph_time: float | None = None,
    ) -> KerrOscillator:
        """
        from_energies Create an AnharmonicOscillator from the charging and Josephson energies. This is so far assuming symmetric junctions.

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
        AnharmonicOscillator
            The AnharmonicOscillator instance.
        """
        if not isinstance(charge_energy, float):
            raise ValueError(
                f"The charge energy must be a float, instead got type {type(charge_energy)}."
            )
        if charge_energy <= 0.0:
            raise ValueError("The charge energy must be greater than zero.")

        if not isinstance(joseph_energy, float):
            raise ValueError(
                f"The Josephson energy must be a float, instead got type {type(joseph_energy)}."
            )
        if joseph_energy <= 0.0:
            raise ValueError("The Josephson energy must be greater than zero.")

        anharmonicity = -charge_energy
        res_freq = math.sqrt(8 * charge_energy * joseph_energy)
        frequency = res_freq + anharmonicity
        return KerrOscillator(
            label=label,
            frequency=frequency,
            anharmonicity=anharmonicity,
            ext_flux=ext_flux,
            asymmetry=asymmetry,
            dim=dim,
            relax_time=relax_time,
            deph_time=deph_time,
        )
