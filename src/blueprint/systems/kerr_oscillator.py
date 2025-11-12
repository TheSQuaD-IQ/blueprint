from __future__ import annotations

from typing import Callable, Tuple


from jax import numpy as jnp
from jaxtyping import Array, Scalar

from .system import System
from ..drives import ChargeDrive, FluxDrive, DetuningDrive
from ..util.linalg import embed_op

type Pulse = Callable[[float], Scalar | Array]


class AnharmonicOscillator(System):
    """AnharmonicOscillator Represents a Kerr (anharmonic) oscillator system."""

    _freq: Array
    _anharm: Array

    def __init__(
        self,
        label: str,
        charging_energy: float | Scalar,
        josephson_energy: float | Scalar,
        dim: int,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> None:
        self.label = str(label)
        self._ec = jnp.asarray(charging_energy)
        self._ej = jnp.asarray(josephson_energy)
        self.dim = int(dim)

        self.drives = {}

        self.device_ind = device_ind
        self.device_dims = device_dims

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
    def charging_energy(self) -> Array:
        """
        charging_energy Returns the charging energy of the transmon.

        Returns
        -------
        Array
            The charging energy of the transmon.
        """
        return self._ec

    @property
    def josephson_energy(self) -> Array:
        """
        josephson_energy Returns the Josephson energy of the transmon.

        Returns
        -------
        Array
            The Josephson energy of the transmon.
        """
        return self._ej

    @property
    def plasma_frequency(self) -> Array:
        """
        plasma_frequency Returns the plasma frequency of the transmon.

        Returns
        -------
        float
            The plasma frequency of the transmon.
        """
        plasma_frequency = jnp.sqrt(8 * self._ec * self._ej)
        return plasma_frequency

    @property
    def frequency(self) -> Array:
        """
        frequency Returns the frequency of the resonator.

        Returns
        -------
        Array
            The frequency of the resonator.
        """
        frequency = jnp.sqrt(8 * self._ec * self._ej) - self._ec
        return frequency

    @property
    def anharmonicity(self) -> Array:
        """
        anharmonicity Returns the anharmonicity of the resonator.

        Returns
        -------
        float
            The anharmonicity of the resonator.
        """
        anharmonicity = -self._ec
        return anharmonicity

    @property
    def charge_zpf(self) -> Array:
        """
        charge_zpf Returns the zero-point fluctuations of the charge.

        Returns
        -------
        Array
            The zero-point fluctuations of the charge.
        """
        charge_zpf = (self._ej / (32 * self._ec)) ** 0.25
        return charge_zpf

    @property
    def flux_zpf(self) -> Array:
        """
        flux_zpf Returns the zero-point fluctuations of the flux.

        Returns
        -------
        Array
            The zero-point fluctuations of the flux.
        """
        flux_zpf = (2 * self._ec / self._ej) ** 0.25
        return flux_zpf

    def embed(
        self, device_ind: int, device_dims: Tuple[int, ...]
    ) -> AnharmonicOscillator:
        """
        embed Embeds the resonator into a larger Hilbert space.

        Parameters
        ----------
        ind : int
            The index of the resonator in the larger Hilbert space.
        device_dims : Tuple[int]
            The dimension of each quantum system (including this resonator) in the full device.
        """

        embedded_oscillator = AnharmonicOscillator(
            label=self.label,
            josephson_energy=self.josephson_energy,
            charging_energy=self.charging_energy,
            dim=self.dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )

        for label, drive in self.drives.items():
            embedded_oscillator.add_drive(label, drive)

        return embedded_oscillator

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
        if self.is_embedded:
            operator = embed_op(operator, self.device_ind, self.device_dims)

        return operator

    def _get_raise_op(self) -> Array:
        """
        _get_raise_op Returns the raising (creation) operator of the resonator.

        Returns
        -------
        Array
            The raising (creation) operator of the resonator.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self.dim))
        raise_op = jnp.diag(offdiag, k=-1)
        return raise_op

    def get_raise_op(self) -> Array:
        """
        get_creation_op Returns the raising (creation) operator of the resonator.

        Returns
        -------
        Array
            The raising (creation) operator in the current basis of the resonator.
        """
        raise_op = self._get_raise_op()
        return self.process_op(raise_op)

    def _get_low_op(self) -> Array:
        """
        _get_low_op Returns the lowering (annihilaton) operator of the resonator.

        Returns
        -------
        Array
            The lowering (annihilaton) operator of the resonator.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self.dim))
        low_op = jnp.diag(offdiag, k=1)
        return low_op

    def get_low_op(self) -> Array:
        """
        get_low_op Returns the lowering (annihilation) operator of the resonator.

        Returns
        -------
        Array
            The lowering (annihilaton) operator in the current basis of the resonator.
        """
        low_op = self._get_low_op()
        return self.process_op(low_op)

    def _get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the resonator.

        Returns
        -------
        Array
            The number operator of the resonator.
        """
        diag_elems = jnp.arange(self.dim)
        number_op = jnp.diag(diag_elems)
        return number_op

    def get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the resonator.

        Returns
        -------
        Array
            The number operator in the current basis of the resonator.
        """
        number_op = self._get_number_op()
        return self.process_op(number_op)

    def _get_identity_op(self) -> Array:
        """
        get_identity_op Returns the identity operator of the transmon.

        Returns
        -------
        Array
            The identity operator of the transmon.
        """
        id_op = jnp.identity(self.dim)
        return id_op

    def get_identity_op(self) -> Array:
        """
        get_identity_op Returns the identity operator of the transmon.

        Returns
        -------
        Array
            The identity operator of the transmon.
        """
        id_op = self._get_identity_op()
        return self.process_op(id_op)

    def _get_charge_op(self) -> Array:
        """
        _get_charge_op Returns the charge operator of the resonator in the Fock basis.

        Returns
        -------
        Array
            The charge operator of the resonator, expressed in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        charge_op = 1.0j * self.charge_zpf * (raise_op - low_op)
        return charge_op

    def get_charge_op(self) -> Array:
        """
        get_charge_op Returns the charge operator of the resonator.

        Returns
        -------
        Array
            The charge operator, in the current basis of the resonator.
        """
        charge_op = self._get_charge_op()
        processed_op = self.process_op(charge_op)
        return processed_op

    def _get_flux_op(self) -> Array:
        """
        _get_flux_op Returns the flux operator of the resonator in the Fock basis.

        Returns
        -------
        Array
            The flux operator of the resonator, expressed in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        flux_op = self.flux_zpf * (raise_op + low_op)
        return flux_op

    def get_flux_op(self) -> Array:
        """
        get_flux_op Returns the flux operator of the resonator.

        Returns
        -------
        Array
            The flux operator, in the current basis of the resonator.
        """
        charge_op = self._get_flux_op()
        processed_op = self.process_op(charge_op)
        return processed_op

    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Returns the Hamiltonian of the fluxonium.

        Returns
        -------
        Array
            The Hamiltonian of the fluxonium.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        number_op = self._get_number_op()

        oscillator_term = self.frequency * number_op

        anharm_op = raise_op @ raise_op @ low_op @ low_op
        anharmonic_term = 0.5 * self._ec * anharm_op

        hamiltonian = oscillator_term - anharmonic_term
        return hamiltonian

    def get_hamiltonian(self) -> Array:
        hamiltonian = self._get_hamiltonian()
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

    def add_detuning_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_detuning_drive Adds a detuning drive to the resonator.

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
        drive = DetuningDrive(label=label, pulse=pulse)
        self.drives[label] = drive
