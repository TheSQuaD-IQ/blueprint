"""Fluxonium qubit model module."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple, Iterable

import yaml
from jax import numpy as jnp
from jax import scipy as jsp
from jaxtyping import Array, Scalar, ScalarLike

from equinox import Module, field

from .system import System
from ..drives import ChargeDrive, FluxDrive
from ..util.linalg import cosm, sinm, transform_op

type Filestring = str | Path
type Pulse = Callable[[float], Scalar | Array]


class FluxoniumParameters(Module):
    """Dataclass for fluxonium parameter storage."""

    label: str = field(static=True)
    charging_energy: Scalar
    inductive_energy: Scalar
    josephson_energy: Scalar
    external_flux: Scalar
    harmonic_cutoff: int = field(static=True)
    dim: int = field(static=True)

    device_ind: int | None = field(static=True)
    device_dims: Tuple[int, ...] | None = field(static=True)

    def __init__(
        self,
        label: str,
        charging_energy: ScalarLike,
        inductive_energy: ScalarLike,
        josephson_energy: ScalarLike,
        external_flux: ScalarLike,
        harmonic_cutoff: int,
        dim: int,
        device_ind: int | None = None,
        device_dims: Iterable[int] | None = None,
    ) -> None:
        """
        __init__ Initialize FluxoniumParameters instance.

        Parameters
        ----------
        label : str
            System label.
        charging_energy, inductive_energy, josephson_energy : ScalarLike
            Fluxonium energy parameters.
        external_flux : ScalarLike
            External magnetic flux through loop.
        harmonic_cutoff : int
            Harmonic cutoff used for native oscillator basis.
        dim : int
            Hilbert-space dimension for returned operators/eigenstates.
        device_ind : int or None, optional
            Embedding index if part of a device.
        device_dims : tuple or None, optional
            Device subsystem dimensions if embedding.
        """
        if not isinstance(label, str):
            raise TypeError("label must be a string.")
        self.label = label

        self.charging_energy = jnp.array(charging_energy)
        self.inductive_energy = jnp.array(inductive_energy)
        self.josephson_energy = jnp.array(josephson_energy)
        self.external_flux = jnp.array(external_flux)

        if not isinstance(harmonic_cutoff, int) or harmonic_cutoff <= 0:
            raise ValueError("harmonic_cutoff must be a positive integer.")
        self.harmonic_cutoff = harmonic_cutoff

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer.")
        self.dim = dim

        if device_ind is not None:
            if not isinstance(device_ind, int):
                raise TypeError("device_ind must be an integer.")

            if device_dims is None:
                raise ValueError(
                    "To embed a system, both device_ind and device_dims must be provided."
                )
        self.device_ind = device_ind

        if device_dims is not None:
            try:
                device_dims = tuple(device_dims)
            except TypeError:
                raise TypeError("device_dims must be an iterable of integers.")

            for dim in device_dims:
                if not isinstance(dim, int) or dim <= 0:
                    raise ValueError(
                        "All entries in device_dims must be positive integers."
                    )

            if device_ind is None:
                raise ValueError(
                    "To embed a system, both device_ind and device_dims must be provided."
                )
            else:
                if device_dims[device_ind] != self.dim:
                    raise ValueError(
                        "The resonator dim must match the corresponding device_dims entry."
                    )

        self.device_dims = device_dims


class Fluxonium(System):
    """Fluxonium qubit system implementation."""

    _ec: Scalar
    _el: Scalar
    _ej: Scalar
    _ext_flux: Scalar
    _hcut: int = field(static=True)

    _eig_vals: Array
    _eig_states: Array

    def __init__(
        self,
        label: str,
        charging_energy: ScalarLike,
        inductive_energy: ScalarLike,
        josephson_energy: ScalarLike,
        ext_flux: ScalarLike,
        harmonic_cutoff: int,
        dim: int,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> None:
        """
        __init__ Initialize a Fluxonium instance.

        Parameters
        ----------
        label : str
            System label.
        charging_energy, inductive_energy, josephson_energy : float or Array
            Fluxonium energy parameters.
        ext_flux : float or Array
            External magnetic flux through loop.
        harmonic_cutoff : int
            Harmonic cutoff used for native oscillator basis.
        dim : int
            Hilbert-space dimension for returned operators/eigenstates.
        device_ind : int or None, optional
            Embedding index if part of a device.
        device_dims : tuple or None, optional
            Device subsystem dimensions if embedding.
        """
        super().__init__(label, dim, device_ind, device_dims)

        self._ej = jnp.array(josephson_energy)
        self._ec = jnp.array(charging_energy)
        self._el = jnp.array(inductive_energy)
        self._ext_flux = jnp.array(ext_flux)

        if not isinstance(harmonic_cutoff, int) or harmonic_cutoff <= 0:
            raise ValueError("harmonic_cutoff must be a positive integer.")
        self._hcut = harmonic_cutoff

        eig_vals, eig_states = self._get_eigenstates()
        self._eig_vals = eig_vals[..., : self.dim]
        self._eig_states = eig_states[..., : self.dim]

    @property
    def charging_energy(self) -> Scalar:
        """
        charging_energy Charging energy parameter E_C for fluxonium.

        Returns
        -------
        Scalar
            Charging energy.
        """
        return self._ec

    @property
    def josephson_energy(self) -> Scalar:
        """
        josephson_energy Josephson energy parameter E_J for fluxonium.

        Returns
        -------
        Scalar
            Josephson energy.
        """
        return self._ej

    @property
    def inductive_energy(self) -> Scalar:
        """
        inductive_energy Inductive energy parameter for fluxonium.

        Returns
        -------
        Scalar
            Inductive energy.
        """
        return self._el

    @property
    def external_flux(self) -> Scalar:
        """
        external_flux External flux threading the fluxonium loop.

        Returns
        -------
        Scalar
            External flux value.
        """
        return self._ext_flux

    @property
    def harmonic_cutoff(self) -> int:
        """
        harmonic_cutoff Harmonic basis cutoff used for native oscillator representation.

        Returns
        -------
        int
            Harmonic cutoff integer.
        """
        return self._hcut

    @property
    def is_diagonal(self) -> bool:
        """
        is_diagonal Whether the fluxonium is represented in its eigenbasis.

        Returns
        -------
        bool
            True when eigenbasis representation is used.
        """
        return True

    @property
    def plasma_frequency(self) -> Scalar:
        """
        plasma_frequency Plasma frequency sqrt(8 E_C E_L) for the harmonic approximation.

        Returns
        -------
        Scalar
            Plasma frequency value.
        """
        return jnp.sqrt(8 * self._ec * self._el)

    @property
    def charge_zpf(self) -> Scalar:
        """
        charge_zpfHarmonic-oscillator charge zero-point fluctuation.

        Returns
        -------
        Scalar
            Charge ZPF.
        """
        return (self._el / (32 * self._ec)) ** 0.25

    @property
    def flux_zpf(self) -> Scalar:
        """
        flux_zpf Harmonic-oscillator flux zero-point fluctuation.

        Returns
        -------
        Scalar
            Flux ZPF.
        """
        return (2 * self._ec / self._el) ** 0.25

    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Fluxonium:
        """
        embed Embed fluxonium into a larger device Hilbert space.

        Parameters
        ----------
        device_ind : int
            Index of this system within the device.
        device_dims : tuple
            Device subsystem dimensions.
        """

        embedded_fluxonium = self.__class__(
            self.label,
            self.charging_energy,
            self.inductive_energy,
            self.josephson_energy,
            self.external_flux,
            self.harmonic_cutoff,
            self.dim,
            device_ind,
            device_dims,
        )

        for label, drive in self.drives.items():
            embedded_fluxonium.drives[label] = drive

        return embedded_fluxonium

    def process_op(self, operator: Array) -> Array:
        """
        process_op Process an operator (transform to energy basis and embed if needed).

        Parameters
        ----------
        operator : Array
            Operator in native basis.

        Returns
        -------
        Array
            Operator in current system representation.
        """
        transformed_op = transform_op(operator, self._eig_states)
        return self.embed_op(transformed_op)

    def _get_raise_op(self) -> Array:
        """
        _get_raise_op Construct raising operator for native harmonic basis.

        Returns
        -------
        Array
            Creation operator matrix.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self._hcut))
        raise_op = jnp.diag(offdiag, k=-1)
        return raise_op

    def get_raise_op(self) -> Array:
        """
        get_raise_op Return raising operator in the fluxonium's current representation.

        Returns
        -------
        Array
            Raising operator in current basis.
        """
        raise_op = self._get_raise_op()
        return self.process_op(raise_op)

    def _get_low_op(self) -> Array:
        """
        _get_low_op Construct lowering operator for native harmonic basis.

        Returns
        -------
        Array
            Lowering operator matrix.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self._hcut))
        low_op = jnp.diag(offdiag, k=1)
        return low_op

    def get_low_op(self) -> Array:
        """
        get_low_op Return lowering operator in the fluxonium's current representation.

        Returns
        -------
        Array
            Lowering operator in current basis.
        """
        low_op = self._get_low_op()
        return self.process_op(low_op)

    def _get_number_op(self) -> Array:
        """
        _get_number_op Construct native number operator for harmonic basis.

        Returns
        -------
        Array
            Number operator matrix.
        """
        diag_elems = jnp.arange(self._hcut)
        num_op = jnp.diag(diag_elems)
        return num_op

    def get_number_op(self) -> Array:
        """
        get_number_op Return number operator in current representation (embedded to `dim`).

        Returns
        -------
        Array
            Number operator in current basis.
        """
        diag_elems = jnp.arange(self.dim)
        num_op = jnp.diag(diag_elems)
        return self.embed_op(num_op)

    def _get_charge_op(self) -> Array:
        """
        _get_charge_op Construct native charge operator from raising/lowering ops.

        Returns
        -------
        Array
            Charge operator in native basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        charge_op = 1.0j * self.charge_zpf * (raise_op - low_op)
        return charge_op

    def get_charge_op(self) -> Array:
        """
        get_charge_op Return charge operator in current representation.

        Returns
        -------
        Array
            Charge operator in current basis.
        """
        charge_op = self._get_charge_op()
        processed_op = self.process_op(charge_op)
        return processed_op

    def _get_flux_op(self) -> Array:
        """
        _get_flux_op Construct native flux operator from raising/lowering ops.

        Returns
        -------
        Array
            Flux operator in native basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        flux_op = self.flux_zpf * (raise_op + low_op)
        return flux_op

    def get_flux_op(self) -> Array:
        """
        get_flux_op Return flux operator in current representation.

        Returns
        -------
        Array
            Flux operator in current basis.
        """
        flux_op = self._get_flux_op()
        processed_op = self.process_op(flux_op)
        return processed_op

    def _get_cosflux_op(self) -> Array:
        """
        _get_cosflux_op Construct cos(flux) operator in native Fock basis.

        Returns
        -------
        Array
            cos(flux) operator matrix.
        """
        flux_op = self._get_flux_op()
        cosflux_op = cosm(flux_op)
        return cosflux_op

    def get_cosflux_op(self) -> Array:
        """
        get_cosflux_op Return cos(flux) operator in current representation.

        Returns
        -------
        Array
            cos(flux) operator in current basis.
        """
        cosflux_op = self._get_cosflux_op()
        processed_op = self.process_op(cosflux_op)
        return processed_op

    def _get_sinflux_op(self) -> Array:
        """
        _get_sinflux_op Construct sin(flux) operator in native Fock basis.

        Returns
        -------
        Array
            sin(flux) operator matrix.
        """
        flux_op = self._get_flux_op()
        sinflux_op = sinm(flux_op)
        return sinflux_op

    def get_sinflux_op(self) -> Array:
        """
        get_sinflux_op Return sin(flux) operator in current representation.

        Returns
        -------
        Array
            sin(flux) operator in current basis.
        """
        sinflux_op = self._get_sinflux_op()
        processed_op = self.process_op(sinflux_op)
        return processed_op

    def _get_identity_op(self) -> Array:
        """
        _get_identity_op Return identity operator for native harmonic cutoff dimension.

        Returns
        -------
        Array
            Identity matrix.
        """
        id_op = jnp.identity(self._hcut)
        return id_op

    def get_identity_op(self) -> Array:
        """
        get_identity_op Return identity operator embedded to the configured `dim`.

        Returns
        -------
        Array
            Identity operator in current representation.
        """
        id_op = jnp.identity(self.dim)
        return self.embed_op(id_op)

    def get_transition_op(self, start_ind: int, end_ind: int) -> Array:
        """
        get_transition_op Return an outer-product transition operator between two levels.

        Parameters
        ----------
        start_ind, end_ind : int
            Level indices for transition ``|end><start|``.

        Returns
        -------
        Array
            Transition operator embedded to device dimension.
        """
        eig_states = jnp.identity(self.dim)

        start_state = eig_states[:, start_ind]
        end_state = eig_states[:, end_ind]
        transition_op = jnp.outer(end_state, start_state)
        return self.embed_op(transition_op)

    def _get_kinetic_term(self) -> Array:
        """
        _get_kinetic_term Construct kinetic term for fluxonium in native Fock basis.

        Returns
        -------
        Array
            Kinetic term matrix.
        """

        n_op = self._get_charge_op()
        kinetic_term = 4 * self._ec * n_op @ n_op
        return kinetic_term

    def _get_potential_term(self) -> Array:
        """
        _get_potential_term Construct potential term for fluxonium in native Fock basis.

        Returns
        -------
        Array
            Potential term matrix.
        """
        cosflux_op = self._get_cosflux_op()
        flux_op = self._get_flux_op()
        id_op = self._get_identity_op()

        offset_flux_op = flux_op + self._ext_flux * id_op

        inductive_term = 0.5 * self._el * (offset_flux_op @ offset_flux_op)
        josephson_term = -self._ej * cosflux_op

        potential_term = inductive_term + josephson_term
        return potential_term

    def _get_oscillator_term(self) -> Array:
        id_op = self._get_identity_op()
        number_op = self._get_number_op()
        oscillator_term = self.plasma_frequency * (number_op + 0.5 * id_op)
        return oscillator_term

    def _get_hamiltonian(self) -> Array:
        """_get_hamiltonian Construct the native fluxonium Hamiltonian (kinetic + potential).

        Returns
        -------
        Array
            Hamiltonian matrix in native Fock basis.
        """
        oscillator_term = self._get_oscillator_term()

        flux_op = self._get_flux_op()
        id_op = self._get_identity_op()
        josephson_term = -self._ej * cosm(flux_op - self._ext_flux * id_op)

        hamiltonian = oscillator_term + josephson_term

        return hamiltonian

    def get_hamiltonian(self) -> Array:
        hamiltonian = jnp.diag(self._eig_vals)
        return self.embed_op(hamiltonian)

    def _get_eigenvalues(self) -> Array:
        """
        _get_eigenvalues Return eigenvalues of native Hamiltonian (ground energy offset).

        Returns
        -------
        Array
            Eigenvalues with ground state set to zero.
        """
        hamiltonian = self._get_hamiltonian()
        eig_vals = jsp.linalg.eigh(hamiltonian, eigvals_only=True)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals

    def get_eigenvalues(self) -> Array:
        return self._eig_vals

    def _get_eigenstates(self) -> Tuple[Array, Array]:
        """
        _get_eigenstates Return eigenvalues and eigenvectors of native Hamiltonian.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of (eigenvalues, eigenvectors).
        """
        hamiltonian = self._get_hamiltonian()
        eig_vals, eig_states = jsp.linalg.eigh(hamiltonian)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals, eig_states

    def get_eigenstates(self) -> Tuple[Array, Array]:
        eig_states = jnp.identity(self.dim, dtype=complex)
        return self._eig_vals, eig_states

    def add_charge_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_charge_drive Attach a charge drive to the fluxonium."""
        drive = ChargeDrive(label, pulse)
        self.drives[label] = drive

    def add_flux_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_flux_drive Attach a flux drive to the fluxonium."""
        drive = FluxDrive(label, pulse)
        self.drives[label] = drive

    @classmethod
    def from_params(cls, params: FluxoniumParameters) -> Fluxonium:
        """
        from_params Construct a Fluxonium instance from a `FluxoniumParameters` object.

        Parameters
        ----------
        params : FluxoniumParameters
            Parameter container for the fluxonium.

        Returns
        -------
        Fluxonium
            Constructed qubit instance.
        """
        fluxonium = cls(
            params.label,
            params.charging_energy,
            params.inductive_energy,
            params.josephson_energy,
            params.external_flux,
            params.harmonic_cutoff,
            params.dim,
            params.device_ind,
            params.device_dims,
        )
        return fluxonium

    @classmethod
    def from_yaml(cls, filename: Filestring) -> Fluxonium:
        """
        from_yaml Load Fluxonium parameters from a YAML file and construct instance.

        The YAML file must contain required fluxonium fields (label, energies,
        cutoff, dim, etc.). Drives are not loaded and must be attached separately.

        Parameters
        ----------
        filename : str or Path
            Path to YAML file containing fluxonium parameters.

        Returns
        -------
        Fluxonium
            Constructed fluxonium instance.
        """
        with open(filename, mode="r", encoding="utf-8") as file:
            parameters = yaml.safe_load(file)

        qubit = cls(**parameters)
        return qubit
