"""Fluxonium qubit module."""

from pathlib import Path
from typing import Callable, Self, Tuple

import yaml
from jax import numpy as jnp
from jax import scipy as jsp
from jaxtyping import Scalar, Array, ArrayLike

from equinox import Module, field

from .system import System
from ..drives import ChargeDrive, FluxDrive
from ..util.linalg import cosm, transform_op

type Filestring = str | Path
type Pulse = Callable[[float], Scalar | Array]


class FluxoniumParameters(Module):
    """Dataclass for storing the parameters of a fluxonium."""

    label: str = field(static=True)
    charging_energy: ArrayLike
    inductive_energy: ArrayLike
    josephson_energy: ArrayLike
    external_flux: float = field(static=True)
    harmonic_cutoff: int = field(static=True)
    dim: int = field(static=True)

    device_ind: int | None = field(static=True)
    device_dims: Tuple[int, ...] | None = field(static=True)


class Fluxonium(System):
    """Fluxonium class for the fluxonium qubit."""

    _ec: Array
    _el: Array
    _ej: Array
    _ext_flux: Array
    _hcut: int = field(static=True)

    _eig_vals: Array
    _eig_states: Array

    def __init__(
        self,
        label: str,
        charging_energy: ArrayLike,
        inductive_energy: ArrayLike,
        josephson_energy: ArrayLike,
        ext_flux: ArrayLike,
        harmonic_cutoff: int,
        dim: int,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> None:
        """
        __init__ Initializes the fluxonium qubit.

        Parameters
        ----------
        josephson_energy : float | Array
            The Josephson energy of the fluxonium.
        charging_energy : float | Array
            The charging energy of the fluxonium.
        inductive_energy : float | Array
            The inductive energy of the fluxonium.
        ext_flux : float | Array
            The external flux applied though the loop of the fluxonium.
        dim : int
            The dimension of the Hilbert space to consider when expressing the fluxonium inm the
            fock basis.
        """
        self.label = str(label)

        self._ej = jnp.asarray(josephson_energy)
        self._ec = jnp.asarray(charging_energy)
        self._el = jnp.asarray(inductive_energy)
        self._ext_flux = jnp.asarray(ext_flux)

        self._hcut = int(harmonic_cutoff)
        self.dim = int(dim)

        self.drives = {}

        self.device_ind = device_ind
        self.device_dims = device_dims

        eig_vals, eig_states = self._get_eigenstates()
        self._eig_vals = eig_vals[..., : self.dim]
        self._eig_states = eig_states[..., : self.dim]

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

    @property
    def harmonic_cutoff(self) -> int:
        """
        harmonic_cutoff Returns the harmonic cutoff of the fluxonium.

        Returns
        -------
        int
            The harmonic cutoff of the fluxonium.
        """
        return self._hcut

    @property
    def is_diagonal(self) -> bool:
        """
        is_diagonal Returns whether the fluxonium is diagonalized.

        Returns
        -------
        bool
            Whether the fluxonium is diagonalized.
        """
        return True

    @property
    def plasma_frequency(self) -> float:
        """
        plasma_frequency Returns the plasma frequency of the fluxonium.

        Returns
        -------
        float
            The plasma frequency of the fluxonium.
        """
        return jnp.sqrt(8 * self._ec * self._el)

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

    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Self:
        """
        embed Embeds the fluxonium into a larger Hilbert space.

        Parameters
        ----------
        ind : int
            The index of the fluxonium in the larger Hilbert space.
        device_dims : Tuple[int]
            The dimension of each quantum system (including this fluxonium)
            in the full device.
        """

        embedded_fluxonium = Fluxonium(
            label=self.label,
            charging_energy=self.charging_energy,
            inductive_energy=self.inductive_energy,
            josephson_energy=self.josephson_energy,
            ext_flux=self.external_flux,
            harmonic_cutoff=self.harmonic_cutoff,
            dim=self.dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )

        for label, drive in self.drives.items():
            embedded_fluxonium.add_drive(label, drive)

        return embedded_fluxonium

    def process_op(self, operator: Array) -> Array:
        """
        process_op Processes an operator of the transmon.
        This includes diagonalizing the operator for operators in the charge basis,
        and embedding it in a larger Hilbert space.

        Parameters
        ----------
        operator : Array
            The operator to process.
        diagonalize : bool, optional
            Whether to transform the operator to the energy eigenbasis of the transmon, by default True
        embed : bool, optional
            Whether to embed the operator in a larger Hilbert space , by default True

        Returns
        -------
        Array
            The processed operator.
        """
        transformed_op = transform_op(operator, self._eig_states)
        return self.embed_op(transformed_op)

    def _get_raise_op(self) -> Array:
        """
        _get_raise_op Returns the raising (creation) operator of the fluxonium.

        Returns
        -------
        Array
            The raising operator, in the Fock basis.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self._hcut))
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
        offdiag = jnp.sqrt(jnp.arange(1, self._hcut))
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
        diag_elems = jnp.arange(self._hcut)
        return jnp.diag(diag_elems)

    def get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the fluxonium.

        Returns
        -------
        Array
            The number operator of the fluxonium.
        """
        diag_elems = jnp.arange(self.dim)
        num_op = jnp.diag(diag_elems)
        return self.embed_op(num_op)

    def _get_charge_op(self) -> Array:
        """
        _get_number_op Returns the number operator of the fluxonium.

        Returns
        -------
        Array
            The number operator of the fluxonium.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        charge_op = 1.0j * self.charge_zpf * (raise_op - low_op)
        return charge_op

    def get_charge_op(self) -> Array:
        """
        get_charge_op Returns the charge operator of the fluxonium.

        Returns
        -------
        Array
            The charge operator, in the current basis of the fluxonium.
        """
        charge_op = self._get_charge_op()
        processed_op = self.process_op(charge_op)
        return processed_op

    def _get_flux_op(self) -> Array:
        """
        _get_charge_op Returns the flux operator of the fluxonium.

        Returns
        -------
        Array
            The flux operator, in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        flux_op = self.flux_zpf * (raise_op + low_op)
        return flux_op

    def get_flux_op(self) -> Array:
        """
        get_flux_op Returns the flux operator of the fluxonium.

        Returns
        -------
        Array
            The flux operator, in the current basis of the fluxonium.
        """
        flux_op = self._get_flux_op()
        processed_op = self.process_op(flux_op)
        return processed_op

    def _get_cosflux_op(self) -> Array:
        """
        _get_cosflux_op Returns the cos(phi) operator of the fluxonium in the fock basis.

        Returns
        -------
        Array
            The cos(phi) operator of the fluxonium, in the Fock basis.
        """
        flux_op = self._get_flux_op()
        cosflux_op = cosm(flux_op)
        return cosflux_op

    def get_cosflux_op(self) -> Array:
        """
        get_cosflux_op Returns the cos(phi) operator of the fluxonium.

        Returns
        -------
        Array
            The cos(phi) operator of the fluxonium, in the current basis of the fluxonium.
        """
        cosflux_op = self._get_cosflux_op()
        processed_op = self.process_op(cosflux_op)
        return processed_op

    def _get_identity_op(self) -> Array:
        """
        get_identity_op Returns the identity operator of the transmon.

        Returns
        -------
        Array
            The identity operator of the transmon.
        """
        id_op = jnp.identity(self._hcut)
        return id_op

    def get_identity_op(self) -> Array:
        """
        get_identity_op Returns the identity operator of the transmon.

        Returns
        -------
        Array
            The identity operator of the transmon.
        """
        id_op = jnp.identity(self.dim)
        return self.embed_op(id_op)

    def get_transition_op(self, start_ind: int, end_ind: int) -> Array:
        """
        get_collapse_ops Returns the collapse operators of the fluxonium.

        Returns
        -------
        Array
            The collapse operators of the fluxonium.
        """
        eig_states = jnp.identity(self.dim)

        start_state = eig_states[:, start_ind]
        end_state = eig_states[:, end_ind]
        transition_op = jnp.outer(end_state, start_state)
        return self.embed_op(transition_op)

    def _get_kinetic_term(self) -> Array:
        """
        _get_kinetic_term Returns the kinetic term of the fluxonium in the fock basis.

        Returns
        -------
        Array
            The kinetic term of the fluxonium, in the Fock basis.
        """

        n_op = self._get_charge_op()
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
        """
        _get_hamiltonian Returns the Hamiltonian of the fluxonium in the fock basis.

        Returns
        -------
        Array
            The Hamiltonian of the fluxonium, in the Fock basis.
        """
        # id_op = jnp.identity(self._hcut)
        # oscillator_term = self._get_oscillator_term()

        # flux_op = self._get_flux_op(include_fluctuations=True)
        # josephson_term = -self._ej * cosm(flux_op - self._ext_flux * id_op)

        # hamiltonian = oscillator_term + josephson_term

        kinetic_term = self._get_kinetic_term()
        potential_term = self._get_potential_term()
        hamiltonian = kinetic_term + potential_term

        return hamiltonian

    def get_hamiltonian(self) -> Array:
        hamiltonian = jnp.diag(self._eig_vals)
        return self.embed_op(hamiltonian)

    def _get_eigenvalues(self) -> Array:
        """
        _get_eigenvalues Returns the eigenvalues of the Hamiltonian of the quantum system.

        Returns
        -------
        Array
            The eigenvalues of the Hamiltonian.
        """
        hamiltonian = self._get_hamiltonian()
        eig_vals = jsp.linalg.eigh(hamiltonian, eigvals_only=True)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals

    def get_eigenvalues(self) -> Array:
        return self._eig_vals

    def _get_eigenstates(self) -> Tuple[Array, Array]:
        """
        _get_eigenstates Returns the eigenvalues and eigenvectors
        of the Hamiltonian of the quantum system.

        Returns
        -------
        Tuple[Array, Array]
            The eigenvalues and eigenvectors of the Hamilton
        """
        hamiltonian = self._get_hamiltonian()
        eig_vals, eig_states = jsp.linalg.eigh(hamiltonian)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals, eig_states

    def get_eigenstates(self) -> Array:
        eig_states = jnp.identity(self.dim, dtype=complex)
        return self._eig_vals, eig_states

    def add_charge_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_charge_drive Adds a charge drive to the fluxonium.
        """
        drive = ChargeDrive(pulse)
        self.drives[label] = drive

    def add_flux_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_flux_drive Adds a flux drive to the fluxonium.
        """
        drive = FluxDrive(pulse)
        self.drives[label] = drive

    @classmethod
    def from_params(cls, parameters: FluxoniumParameters) -> Self:
        """
        from_params Initializes a fluxonium from a set of parameters.

        Parameters
        ----------
        parameters : FluxoniumParameters
            The parameters of the fluxonium.

        Returns
        -------
        Self
            The fluxonium initialized from the parameters.
        """
        qubit = cls(
            label=parameters.label,
            charging_energy=parameters.charging_energy,
            inductive_energy=parameters.inductive_energy,
            josephson_energy=parameters.josephson_energy,
            ext_flux=parameters.external_flux,
            harmonic_cutoff=parameters.harmonic_cutoff,
            dim=parameters.dim,
            device_ind=parameters.device_ind,
            device_dims=parameters.device_dims,
        )
        return qubit

    @classmethod
    def from_yaml(cls, filename: Filestring) -> Self:
        """
        from_yaml Initializes a fluxonium from a YAML file
        which defines the parameters of the fluxonium.
        The YAML file should contain the following fields:
        - label: str
        - charging_energy: float
        - inductive_energy: float
        - josephson_energy: float
        - ext_flux: float
        - harmonic_cutoff: int
        - dim: int
        - device_ind: Optional[int]
        - device_dims: Optional[Tuple[int, ...]]

        Note that the drives acting on the qubit must be initialized seperately.

        Parameters
        ----------
        filename : Filestring
            The path to the YAML file containing the qubit parameters.
            This can be provided either as a string or a pathlib.Path object.

        Returns
        -------
        Self
            The fluxonium initialized from the YAML file.

        """
        with open(filename, mode="r", encoding="utf-8") as file:
            parameters = yaml.safe_load(file)

        qubit = cls(**parameters)
        return qubit
