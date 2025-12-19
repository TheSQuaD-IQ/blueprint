from __future__ import annotations

import math
from typing import Callable, Self, Tuple

from scipy.constants import e, hbar

from jax import numpy as jnp
from jaxtyping import ArrayLike, Array, Scalar
from equinox import Module, field

from .system import System
from ..drives import ChargeDrive, FluxDrive
from ..util.linalg import embed_op

type Pulse = Callable[[float], Scalar | Array]


class ResonatorParams(Module):
    """Parameters container for a resonator (helper dataclass)."""

    label: str = field(static=True)
    charging_energy: ArrayLike
    inductive_energy: ArrayLike
    dim: int = field(static=True)

    device_ind: int | None = field(default=None, static=True)
    device_dims: Tuple[int, ...] | None = field(default=None, static=True)

    @staticmethod
    def from_frequency(
        label: str,
        frequency: float,
        impedance: float,
        dim: int,
        *,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> ResonatorParams:
        """
        from_frequency Create ResonatorParams from frequency and impedance.

        Parameters
        ----------
        label : str
            Resonator label.
        frequency : float
            Resonator frequency.
        impedance : float
            Characteristic impedance.
        dim : int
            Hilbert-space dimension to model.
        device_ind : int or None, optional
            Device embedding index if applicable.
        device_dims : tuple or None, optional
            Device subsystem dimensions if embedding.

        Returns
        -------
        ResonatorParams
            Constructed parameters object.
        """
        capacitance = 1 / (impedance * frequency)

        redifined_e = e / math.sqrt(hbar)
        charging_energy = (redifined_e**2) / (2 * capacitance)

        inductance = impedance / frequency
        inductive_energy = 1 / (4 * (redifined_e**2) * inductance)

        params = ResonatorParams(
            label=label,
            charging_energy=charging_energy,
            inductive_energy=inductive_energy,
            dim=dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )
        return params


class Resonator(System):
    """Resonator model implementation."""

    _ec: Array
    _el: Array

    def __init__(
        self,
        label: str,
        charging_energy: ArrayLike,
        inductive_energy: ArrayLike,
        dim: int,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> None:
        self.label = str(label)
        self._ec = jnp.asarray(charging_energy)
        self._el = jnp.asarray(inductive_energy)
        self.dim = int(dim)

        self.drives = {}

        self.device_ind = device_ind
        self.device_dims = device_dims

    @property
    def charging_energy(self) -> float:
        """
        charging_energy Charging energy parameter E_C for the resonator.

        Returns
        -------
        float
            Charging energy value.
        """
        return self._ec

    @property
    def inductive_energy(self) -> float:
        """
        inductive_energy Inductive energy parameter for the resonator.

        Returns
        -------
        float
            Inductive energy value.
        """
        return self._el

    @property
    def is_diagonal(self) -> bool:
        """
        is_diagonal Whether the resonator representation is diagonal in its native basis.

        Returns
        -------
        bool
            True if represented in energy eigenbasis.
        """
        return True

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
    def charge_zpf(self) -> float:
        """
        charge_zpf Charge zero-point fluctuations for the resonator.

        Returns
        -------
        float
            Charge zero-point fluctuation.
        """
        return (self._el / (32 * self._ec)) ** 0.25

    @property
    def flux_zpf(self) -> float:
        """
        flux_zpf Flux zero-point fluctuations for the resonator.

        Returns
        -------
        float
            Flux zero-point fluctuation.
        """
        return (2 * self._ec / self._el) ** 0.25

    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Self:
        """
        embed Embed resonator into a larger device Hilbert space.

        Parameters
        ----------
        device_ind : int
            Embedding index of this resonator.
        device_dims : tuple
            Subsystem dimensions for the device.
        """

        embedded_resonator = Resonator(
            label=self.label,
            charging_energy=self.charging_energy,
            inductive_energy=self.inductive_energy,
            dim=self.dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )

        for label, drive in self.drives.items():
            embedded_resonator.drives[label] = drive

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
        if self.is_embedded:
            operator = embed_op(operator, self.device_ind, self.device_dims)

        return operator

    def _get_raise_op(self) -> Array:
        """
        _get_raise_op Construct raising (creation) operator in Fock basis.

        Returns
        -------
        Array
            Creation operator matrix.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self.dim))
        raise_op = jnp.diag(offdiag, k=-1)
        return raise_op

    def get_raise_op(self) -> Array:
        """
        get_raise_op Return raising operator in the system's current representation.

        Returns
        -------
        Array
            Raising operator in current basis.
        """
        raise_op = self._get_raise_op()
        return self.process_op(raise_op)

    def _get_low_op(self) -> Array:
        """
        _get_low_op Construct lowering (annihilation) operator in Fock basis.

        Returns
        -------
        Array
            Lowering operator matrix.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self.dim))
        low_op = jnp.diag(offdiag, k=1)
        return low_op

    def get_low_op(self) -> Array:
        """
        get_low_op Return lowering operator in the system's current representation.

        Returns
        -------
        Array
            Lowering operator in current basis.
        """
        low_op = self._get_low_op()
        return self.process_op(low_op)

    def _get_number_op(self) -> Array:
        """
        _get_number_op Construct number operator in Fock basis.

        Returns
        -------
        Array
            Number operator matrix.
        """
        diag_elems = jnp.arange(self.dim)
        number_op = jnp.diag(diag_elems)
        return number_op

    def get_number_op(self) -> Array:
        """
        get_number_op Return number operator in the current representation.

        Returns
        -------
        Array
            Number operator in current basis.
        """
        number_op = self._get_number_op()
        return self.process_op(number_op)

    def _get_identity_op(self) -> Array:
        """
        _get_identity_op Return identity operator in native Fock basis.

        Returns
        -------
        Array
            Identity matrix.
        """
        id_op = jnp.identity(self.dim)
        return id_op

    def get_identity_op(self) -> Array:
        """
        get_identity_op Return identity operator in current representation.

        Returns
        -------
        Array
            Identity operator in current basis.
        """
        id_op = self._get_identity_op()
        return self.process_op(id_op)

    def _get_charge_op(self) -> Array:
        """
        _get_charge_op Construct native charge operator in Fock basis.

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
        _get_flux_op Construct native flux operator in Fock basis.

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
        charge_op = self._get_flux_op()
        processed_op = self.process_op(charge_op)
        return processed_op

    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Construct native Hamiltonian (simple harmonic oscillator approximation).

        Returns
        -------
        Array
            Hamiltonian matrix in native basis.
        """
        number_op = self._get_number_op()
        hamiltonian = self.plasma_frequency * number_op
        return hamiltonian

    def get_hamiltonian(self) -> Array:
        hamiltonian = self._get_hamiltonian()
        return self.process_op(hamiltonian)

    def get_eigenvalues(self) -> Array:
        prefactors = jnp.arange(self.dim)
        eig_vals = self.plasma_frequency * prefactors
        return eig_vals

    def get_eigenstates(self) -> Array:
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
        self.drives[label] = drive

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
        self.drives[label] = drive

    @staticmethod
    def from_frequency(
        label: str,
        frequency: float,
        impedance: float,
        dim: int,
        *,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> Self:
        """
        from_frequency Returns a Resonator object from the frequency and impedence of the resonator.

        Parameters
        ----------
        label : str
            The label of the resonator.
        frequency : float
            The frequency of the resonator.
        impedance : float
            The characteristic impedance of the resonator.
        dim : int, optional
            The dimensionality of the resonator.


        Returns
        -------
        Resonator
            The resulting Resonator .

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
        capacitance = 1 / (impedance * frequency)

        redifined_e = e / math.sqrt(hbar)
        charging_energy = (redifined_e**2) / (2 * capacitance)

        inductance = impedance / frequency
        inductive_energy = 1 / (4 * (redifined_e**2) * inductance)

        oscillator = Resonator(
            label,
            charging_energy,
            inductive_energy,
            dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )
        return oscillator

    @staticmethod
    def from_params(params: ResonatorParams) -> Self:
        """
        from_params Initializes a Resonator object from the given ResonatorParams object.

        Parameters
        ----------
        params : ResonatorParams
            The parameters of the resonator.

        Returns
        -------
        Self
            The resulting Resonator object.
        """
        resonator = Resonator(
            label=params.label,
            charging_energy=params.charging_energy,
            inductive_energy=params.inductive_energy,
            dim=params.dim,
            device_ind=params.device_ind,
            device_dims=params.device_dims,
        )
        return resonator
