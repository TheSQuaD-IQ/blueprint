"""Fluxonium qubit model module."""

from __future__ import annotations

from typing import Callable, Tuple

from jax import numpy as jnp
from jax import scipy as jsp
from jaxtyping import Array, Scalar

from equinox import field

from .system import System
from ..drives import ChargeDrive, FluxDrive
from ..operators import harmonic as harmonic_ops
from ..util.linalg import cosm, transform_op

type Float = float | Scalar
type Pulse = Callable[[Scalar], Array]


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
        charging_energy: Float,
        inductive_energy: Float,
        josephson_energy: Float,
        ext_flux: Float,
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
        charging_energy, inductive_energy, josephson_energy : Float
            Fluxonium energy parameters.
        ext_flux : Float
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

        for label, drive in self._drives.items():
            embedded_fluxonium._drives[label] = drive

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

    def get_raise_op(self) -> Array:
        """
        get_raise_op Return raising operator in the fluxonium's current representation.

        Returns
        -------
        Array
            Raising operator in current basis.
        """
        raise_op = harmonic_ops.get_raise_op(self._hcut)
        return self.process_op(raise_op)

    def get_low_op(self) -> Array:
        """
        get_low_op Return lowering operator in the fluxonium's current representation.

        Returns
        -------
        Array
            Lowering operator in current basis.
        """
        low_op = harmonic_ops.get_low_op(self._hcut)
        return self.process_op(low_op)

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

    def get_charge_op(self) -> Array:
        """
        get_charge_op Return charge operator in current representation.

        Returns
        -------
        Array
            Charge operator in current basis.
        """
        charge_op = harmonic_ops.get_charge_op(self.charge_zpf, self._hcut)
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
        flux_op = harmonic_ops.get_flux_op(self.flux_zpf, self._hcut)
        processed_op = self.process_op(flux_op)
        return processed_op

    def get_cosflux_op(self) -> Array:
        """
        get_cosflux_op Return cos(flux) operator in current representation.

        Returns
        -------
        Array
            cos(flux) operator in current basis.
        """
        cosflux_op = harmonic_ops.get_cosflux_op(self.flux_zpf, self._hcut)
        processed_op = self.process_op(cosflux_op)
        return processed_op

    def get_sinflux_op(self) -> Array:
        """
        get_sinflux_op Return sin(flux) operator in current representation.

        Returns
        -------
        Array
            sin(flux) operator in current basis.
        """
        sinflux_op = harmonic_ops.get_sinflux_op(self.flux_zpf, self._hcut)
        processed_op = self.process_op(sinflux_op)
        return processed_op

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

    def _get_oscillator_term(self) -> Array:
        id_op = harmonic_ops.get_identity_op(self._hcut)
        number_op = harmonic_ops.get_number_op(self._hcut)
        oscillator_term = self.plasma_frequency * (number_op + 0.5 * id_op)
        return oscillator_term

    def _get_josephson_term(self) -> Array:
        flux_op = harmonic_ops.get_flux_op(self.flux_zpf, self._hcut)
        id_op = harmonic_ops.get_identity_op(self._hcut)
        josephson_term = -self._ej * cosm(flux_op - self._ext_flux * id_op)
        return josephson_term

    def _get_hamiltonian(self) -> Array:
        """_get_hamiltonian Construct the native fluxonium Hamiltonian (kinetic + potential).

        Returns
        -------
        Array
            Hamiltonian matrix in native Fock basis.
        """
        oscillator_term = self._get_oscillator_term()
        josephson_term = self._get_josephson_term()
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
        self._drives[label] = drive

    def add_flux_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_flux_drive Attach a flux drive to the fluxonium."""
        drive = FluxDrive(label, pulse)
        self._drives[label] = drive
