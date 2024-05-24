"""Fluxonium qubit module."""

import math
from typing import Callable, Iterator

from jax import Array
from jax import numpy as jnp

from ..base import QuantumSystem
from ..util.linalg import cosm


def check_var_validity(
    arg: float,
    argname: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """
    check_var_validity Checks the validity of a variable.
    Note that this function does not accept value equal to the minimum or maximum value.

    Parameters
    ----------
    arg : float
        The variable to check.
    argname : str
        The name of the variable.
    min_value : float | None, optional
        The minimum value that the variable can take , by default None
    max_value : float | None, optional
        The maximum value the variable can take , by default None

    Raises
    ------
    ValueError
        If the variable is not a float.
    ValueError
        If the variable is less than the minimum value.
    ValueError
        If the variable is greater than the maximum value.
    """
    if not isinstance(arg, float):
        raise ValueError(
            f"The {argname} is expected to be a float, instead got type {type(arg)}."
        )

    if min_value is not None:
        if arg < min_value:
            raise ValueError(
                f"The {argname} must be greater than or equal to {min_value}."
            )
    if max_value is not None:
        if arg > max_value:
            raise ValueError(
                f"The {argname} must be less than or equal to {max_value}."
            )


class Fluxonium(QuantumSystem):
    """
    Fluxonium _summary_

    Parameters
    ----------
    QuantumSystem : _type_
        _description_
    """

    def __init__(
        self,
        label: str,
        josephson_energy: float,
        charging_energy: float,
        inductive_energy: float,
        ext_flux: float,
        fock_cutoff: int,
    ) -> None:
        """
        __init__ Initializes the fluxonium qubit.

        Parameters
        ----------
        label : str
            The label of the fluxonium.
        josephson_energy : float
            The Josephson energy of the fluxonium.
        charging_energy : float
            The charging energy of the fluxonium.
        inductive_energy : float
            The inductive energy of the fluxonium.
        ext_flux : float
            The external flux applied though the loop of the fluxonium.
        fock_cutoff : int
            The number of Fock-basis states to consider when constructing the Hamiltonian/operators.

        Raises
        ------
        ValueError
            If any of the energies are not floats or if they are negative.
        ValueError
            If the external flux is not a float.
        ValueError
            If the Fock-basis cutoff is not an integer or is less than or equal to zero.
        """
        check_var_validity(josephson_energy, "josephson_energy", min_value=0.0)
        self._ej = josephson_energy

        check_var_validity(charging_energy, "charging_energy", min_value=0.0)
        self._ec = charging_energy

        check_var_validity(inductive_energy, "inductive_energy", min_value=0.0)
        self._el = inductive_energy

        check_var_validity(ext_flux, "ext_flux")
        self._ext_flux = ext_flux

        # The number of charge states to consider when constructing the Hamiltonian/operators
        # in the native (charge) basis.
        if not isinstance(fock_cutoff, int):
            raise ValueError(
                f"The Fock-basis cutoff expected to be an integer, "
                f"instead got type {type(fock_cutoff)}."
            )
        if fock_cutoff <= 0:
            raise ValueError(
                "The Fock-basis cutoff must be a non-negative integer or equal to zero."
            )
        self._fcut: int = fock_cutoff

        # The dimension of the Hilbert space
        dim: int = self._fcut
        super().__init__(label, dim)

    @property
    def josephson_energy(self) -> float:
        """
        josephson_energy Returns the josephson energy of the fluxonium.

        Returns:
        -------
        float:
            The josephson energy of the fluxonium.
        """
        return self._ej

    @josephson_energy.setter
    def josephson_energy(self, josephson_energy: float) -> None:
        """
        josephson_energy Sets the inductive energy of the fluxonium.

        Parameters
        ----------
        josephson_energy : float
            The inductive energy of the fluxonium.

        Raises
        ------
        ValueError
            If the inductive energy is not a float.
        ValueError
            If the inductive energy is less than or equal to zero.
        """
        if not isinstance(josephson_energy, float):
            raise ValueError(
                f"The Josephson energy must be a float, instead got type {type(josephson_energy)}."
            )
        if josephson_energy <= 0.0:
            raise ValueError("The Josephson energy must be greater than zero.")
        self._el = josephson_energy

    @property
    def charging_energy(self) -> float:
        """
        charging_energy Returns the josephson energy of the fluxonium.

        Returns:
        -------
        float:
            The charging energy of the fluxonium.
        """
        return self._ec

    @charging_energy.setter
    def charging_energy(self, charging_energy: float) -> None:
        """
        charging_energy Sets the charging energy of the fluxonium.

        Parameters
        ----------
        charging_energy : float
            The charging energy of the fluxonium.

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
        inductive_energy Returns the inductive energy of the fluxonium.

        Returns:
        -------
        float:
            The inductive energy of the fluxonium.
        """
        return self._el

    @inductive_energy.setter
    def inductive_energy(self, inductive_energy: float) -> None:
        """
        inductive_energy Sets the inductive energy of the fluxonium.

        Parameters
        ----------
        inductive_energy : float
            The inductive energy of the fluxonium.

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
    def external_flux(self) -> float:
        """
        external_flux Returns the external flux of the fluxonium.

        Returns
        -------
        float
            The external flux of the fluxonium.
        """
        return self._ext_flux

    @external_flux.setter
    def external_flux(self, ext_flux: float) -> None:
        """
        inductive_energy Sets the inductive energy of the fluxonium.

        Parameters
        ----------
        inductive_energy : float
            The inductive energy of the fluxonium.

        Raises
        ------
        ValueError
            If the inductive energy is not a float.
        ValueError
            If the inductive energy is less than or equal to zero.
        """
        if not isinstance(ext_flux, float):
            raise ValueError(
                f"The external flux must be a float, instead got type {type(ext_flux)}."
            )
        self._ext_flux = ext_flux

    @property
    def fock_cutoff(self) -> int:
        """
        charge_cutoff Returns the number of charge states to consider.

        Returns
        -------
        int
            The number of charge states to consider.
        """
        return self._fcut

    @fock_cutoff.setter
    def fock_cutoff(self, fock_cutoff: int) -> None:
        """
        fock_cutoff Sets the number of Fock-basis states used to represent the fluxonium.

        Parameters
        ----------
        fock_cutoff : int
            The number of Fock-bassi states.

        Raises
        ------
        ValueError
            If the fock_cutoff is not an integer.
        ValueError
            If the fock_cutoff is less than or equal to zero.
        """
        if not isinstance(fock_cutoff, int):
            raise ValueError(
                f"The Fock-basis cutoff expected to be an integer, instead got type {type(fock_cutoff)}."
            )
        if fock_cutoff <= 0:
            raise ValueError(
                "The Fock-basis cutoff must be a non-negative integer or equal to zero."
            )
        self._fcut = fock_cutoff

    @property
    def charge_zpf(self) -> float:
        """
        charge_zpf Returns the zero-point fluctuations of the harmonic charge variable.

        Returns
        -------
        float
            The zero-point fluctuations of the charge.
        """
        return (self._el / (32 * self._ec)) ** 0.25

    @property
    def flux_zpf(self) -> float:
        """
        flux_zpf Returns the zero-point fluctuations of the harmonic flux variable.

        Returns
        -------
        float
            The zero-point fluctuations of the flux.
        """
        return (2 * self._ec / self._el) ** 0.25

    @property
    def plasma_frequency(self) -> float:
        """
        plasma_frequency Returns the plasma frequency of the fluxonium.

        Returns
        -------
        float
            The plasma frequency of the fluxonium.
        """
        return math.sqrt(8 * self._ec * self._el)

    def _get_raise_op(self) -> Array:
        """
        _get_raise_op Returns the raising (creation) operator of the fluxonium.

        Returns
        -------
        Array
            The raising operator, in the Fock basis.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self._dim))
        raise_op = jnp.diag(offdiag, k=-1)
        return raise_op

    def get_raise_op(self) -> Array:
        """
        get_raise_op Returns the raising (creation) operator of the fluxonium.

        Returns
        -------
        Array
            The raising operator, in the current basis of the fluxonium.
        """
        raise_op = self._get_raise_op()
        return self.process_op(raise_op)

    def _get_low_op(self) -> Array:
        """
        _get_low_op Returns the lowering (annihilaton) operator of the fluxonium.

        Returns
        -------
        Array
            The lowering (annihilaton) operator of the fluxonium.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self._dim))
        low_op = jnp.diag(offdiag, k=1)
        return low_op

    def get_low_op(self) -> Array:
        """
        get_low_op Returns the creation operator of the fluxonium.

        Returns
        -------
        Array
            The lowering (annihilaton) operator in the current basis of the fluxonium.
        """
        low_op = self._get_low_op()
        return self.process_op(low_op)

    def _get_number_op(self) -> Array:
        """
        _get_number_op Returns the number operator of the fluxonium.

        Returns
        -------
        Array
            The number operator of the fluxonium.
        """
        diag_elems = jnp.arange(self._dim)
        return jnp.diag(diag_elems)

    def get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the fluxonium.

        Returns
        -------
        Array
            The number operator of the fluxonium.

        Raises
        ------
        NotImplementedError
            If the fluxonium is not diagonalized.
        """
        if self._diagonalized:
            diag_elems = jnp.arange(self._dim)
            number_op = jnp.diag(diag_elems)
            processed_op = self.process_op(number_op, diagonalize=False)
            return processed_op

        raise NotImplementedError(
            "The number operator is only available in the diagonal (energy) basis."
        )

    def _get_charge_op(self, include_fluctuations: bool) -> Array:
        """
        _get_number_op Returns the number operator of the fluxonium.

        Returns
        -------
        Array
            The number operator of the fluxonium.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        charge_op = 1.0j * (raise_op - low_op)
        if include_fluctuations:
            return self.charge_zpf * charge_op
        return charge_op

    def get_charge_op(self, *, include_fluctuations: bool = True) -> Array:
        """
        get_charge_op Returns the charge operator of the fluxonium.

        Returns
        -------
        Array
            The charge operator, in the current basis of the fluxonium.
        """
        charge_op = self._get_charge_op(include_fluctuations)
        processed_op = self.process_op(charge_op)
        return processed_op

    def _get_flux_op(self, include_fluctuations: bool) -> Array:
        """
        _get_charge_op Returns the flux operator of the fluxonium.

        Returns
        -------
        Array
            The flux operator, in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        flux_op = raise_op + low_op
        if include_fluctuations:
            return self.flux_zpf * flux_op
        return flux_op

    def get_flux_op(self, *, include_fluctuations: bool = True) -> Array:
        """
        get_flux_op Returns the flux operator of the fluxonium.

        Returns
        -------
        Array
            The flux operator, in the current basis of the fluxonium.
        """
        flux_op = self._get_flux_op(include_fluctuations)
        processed_op = self.process_op(flux_op)
        return processed_op

    def _get_cosphi_op(self, include_fluctuations: bool) -> Array:
        """
        _get_cosphi_op Returns the cos(phi) operator of the fluxonium in the fock basis.

        Returns
        -------
        Array
            The cos(phi) operator of the fluxonium, in the Fock basis.
        """
        flux_op = self._get_flux_op(include_fluctuations)
        cosphi_op = cosm(flux_op)
        return cosphi_op

    def get_cosphi_op(self, *, include_fluctuations: bool = True) -> Array:
        """
        get_cosphi_op Returns the cos(phi) operator of the fluxonium.

        Returns
        -------
        Array
            The cos(phi) operator of the fluxonium, in the current basis of the fluxonium.
        """
        cosphi_op = self._get_cosphi_op(include_fluctuations)
        processed_op = self.process_op(cosphi_op)
        return processed_op

    def _get_kinetic_term(self) -> Array:
        """
        _get_kinetic_term Returns the kinetic term of the fluxonium in the fock basis.

        Returns
        -------
        Array
            The kinetic term of the fluxonium, in the Fock basis.
        """

        n_op = self._get_charge_op(include_fluctuations=True)
        kinetic_term = 4 * self._ec * n_op @ n_op
        return kinetic_term

    def _get_potential_term(self) -> Array:
        """
        _get_potential_term Returns the potential term of the fluxonium in the fock basis.

        Returns
        -------
        Array
            The potential term of the fluxonium, in the Fock basis.
        """
        cosphi_op = self._get_cosphi_op(include_fluctuations=True)
        flux_op = self._get_flux_op(include_fluctuations=True)
        id_op = jnp.identity(self.dim)

        offset_flux_op = flux_op + self._ext_flux * id_op

        inductive_term = 0.5 * self._el * (offset_flux_op @ offset_flux_op)
        josephson_term = -self._ej * cosphi_op

        potential_term = inductive_term + josephson_term
        return potential_term

    def _get_oscillator_term(self) -> Array:
        id_op = jnp.identity(self._dim)
        number_op = self._get_number_op()
        oscillator_term = self.plasma_frequency * (number_op + 0.5 * id_op)
        return oscillator_term

    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Returns the Hamiltonian of the fluxonium in the fock basis.

        Returns
        -------
        Array
            The Hamiltonian of the fluxonium, in the Fock basis.
        """
        # id_op = jnp.identity(self._dim)
        # oscillator_term = self._get_oscillator_term()

        # flux_op = self._get_flux_op(include_fluctuations=True)
        # josephson_term = -self._ej * cosm(flux_op - self._ext_flux * id_op)

        # hamiltonian = oscillator_term + josephson_term

        kinetic_term = self._get_kinetic_term()
        potential_term = self._get_potential_term()
        hamiltonian = kinetic_term + potential_term

        return hamiltonian

    def get_jump_ops(self) -> Iterator[Array]:
        raise NotImplementedError("Jump operators are not implemented yet.")

    def add_charge_drive(
        self, label: str, charge_pulse: Callable, *, include_fluctuations: bool = True
    ) -> None:
        """
        add_charge_drive Applies a charge drive to the fluxonium.

        Parameters
        ----------
        label : str
            The label of the drive.
        charge_pulse : Callable
            The time-dependent charge pulse applied to the fluxonium.
            This must be a callable object that returns the prefactor in front of the charge operator as a function of the time `t`.
        include_fluctuations : bool, optional
            Whether to include the charge operator fluctuations, by default True
        """
        charge_op = self._get_charge_op(include_fluctuations)
        self.add_drive(label, charge_pulse, charge_op)

    def add_flux_drive(
        self, label: str, flux_pulse: Callable, *, include_fluctuations: bool = True
    ) -> None:
        """
        add_flux_drive Applies a flux drive to the fluxonium.

        Parameters
        ----------
        label : str
            The label of the drive.
        flux_pulse : Callable
            The time-dependent flux pulse applied to the fluxonium.
            This must be a callable object that returns the prefactor in front of the flux operator as a function of the time `t`.
        """
        raise NotImplementedError
