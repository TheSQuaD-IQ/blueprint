from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Tuple, Iterable

from jax import numpy as jnp
from jax import scipy as jsp
from jaxtyping import Scalar, Array

from equinox import field

from optimistix import minimise, BFGS

from .system import System
from ..operators import charge as charge_ops
from ..drives import ChargeDrive, FluxDrive, CosFluxDrive, SinFluxDrive
from ..util.linalg import transform_op

type Float = float | Scalar
type Pulse = Callable[[Scalar], Array]


class BaseTransmon(System):
    """Base class for Transmon-like qubit models."""

    _ec: Scalar
    _ej: Scalar
    _ng: Scalar
    _ncut: int = field(static=True)

    def __init__(
        self,
        label: str,
        charging_energy: Float,
        josephson_energy: Float,
        offset_charge: Float,
        charge_cutoff: int,
        dim: int,
        device_ind: int | None = None,
        device_dims: Iterable[int] | None = None,
    ) -> None:
        super().__init__(label, dim, device_ind, device_dims)
        self._ec = jnp.asarray(charging_energy)
        self._ej = jnp.asarray(josephson_energy)
        self._ng = jnp.asarray(offset_charge)

        if not isinstance(charge_cutoff, int) or charge_cutoff <= 0:
            raise ValueError("charge_cutoff must be a positive integer.")

        self._ncut = charge_cutoff

    @property
    def charging_energy(self) -> Scalar:
        """
        charging_energy Charging energy parameter E_C.

        Returns
        -------
        Scalar
            Charging energy of the transmon.
        """
        return self._ec

    @property
    def josephson_energy(self) -> Scalar:
        """
        josephson_energy Josephson energy parameter E_J.

        Returns
        -------
        Scalar
            Josephson energy of the transmon.
        """
        return self._ej

    @property
    def offset_charge(self) -> Scalar:
        """
        offset_charge Offset charge (n_g) of the transmon.

        Returns
        -------
        Scalar
            Offset charge parameter.
        """
        return self._ng

    @property
    def charge_cutoff(self) -> int:
        """
        charge_cutoff Number of charge basis states retained (n_cut).

        Returns
        -------
        int
            Charge cutoff used in computations.
        """
        return self._ncut

    @property
    def charge_zpf(self) -> Scalar:
        """
        charge_zpfZero-point fluctuations of the charge variable (in energy basis).

        Notes
        -----
        This assumes the system is diagonalized; otherwise values may be
        meaningless.

        Returns
        -------
        Scalar
            Charge zero-point fluctuation.
        """
        charge_fluctuations = (self._ej / (32 * self._ec)) ** 0.25
        return charge_fluctuations

    @property
    def flux_zpf(self) -> Scalar:
        """
        flux_zpf Zero-point fluctuations of the flux variable (in energy basis).

        Returns
        -------
        Scalar
            Flux zero-point fluctuation.
        """
        flux_fluctuations = (2 * self._ec / self._ej) ** 0.25
        return flux_fluctuations

    @abstractmethod
    def process_op(self, operator: Array) -> Array:
        """
        process_op Process an operator into the transmon's current basis/embedding.

        Parameters
        ----------
        operator : Array
            Operator in the native basis.

        Returns
        -------
        Array
            Operator in the system's current representation.
        """

    def get_charge_op(self) -> Array:
        """
        get_charge_op Return the charge operator in the system's current basis.

        Returns
        -------
        Array
            Charge operator in current representation.
        """
        native_op = charge_ops.get_charge_op(self._ng, self._ncut)
        return self.process_op(native_op)

    def get_cosflux_op(self) -> Array:
        """
        get_cosflux_op Return cos(flux) operator in the system's current basis.

        Returns
        -------
        Array
            cos(flux) operator in current representation.
        """
        native_op = charge_ops.get_cosflux_op(self._ncut)
        op = self.process_op(native_op)
        return op

    def get_sinflux_op(self) -> Array:
        """
        get_sinflux_op Return sin(flux) operator in the system's current basis.

        Returns
        -------
        Array
            sin(flux) operator in current representation.
        """
        native_op = charge_ops.get_sinflux_op(self._ncut)
        op = self.process_op(native_op)
        return op

    def get_identity_op(self) -> Array:
        """
        get_identity_op Return identity operator in the system's current basis.

        Returns
        -------
        Array
            Identity operator in current representation.
        """
        id_op = jnp.identity(self.dim)
        return self.embed_op(id_op)

    def _get_kinetic_term(self) -> Array:
        """
        _get_kinetic_term Construct kinetic term of the transmon Hamiltonian in charge basis.

        Returns
        -------
        Array
            Kinetic term matrix in native basis.
        """
        offset_charge_op = charge_ops.get_charge_op(self._ng, self._ncut)
        kinetic_term = 4 * self._ec * offset_charge_op @ offset_charge_op
        return kinetic_term

    def _get_potential_term(self) -> Array:
        """
        _get_potential_term Construct potential (Josephson) term of the Hamiltonian in charge basis.

        Returns
        -------
        Array
            Potential term matrix in native basis.
        """
        cosflux_op = charge_ops.get_cosflux_op(self._ncut)
        potential_term = -self._ej * cosflux_op
        return potential_term

    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Construct full transmon Hamiltonian in the native charge basis.

        Returns
        -------
        Array
            Hamiltonian matrix in native basis.
        """
        kinetic_term = self._get_kinetic_term()
        potential_term = self._get_potential_term()
        hamiltonian = kinetic_term + potential_term
        return hamiltonian

    def _get_eigenvalues(self) -> Array:
        """
        _get_eigenvalues Return eigenvalues of the native Hamiltonian (ground energy offset).

        Returns
        -------
        Array
            Eigenvalues with ground state set to zero.
        """
        hamiltonian = self._get_hamiltonian()
        eig_vals = jsp.linalg.eigh(hamiltonian, eigvals_only=True)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals

    def _get_eigenstates(self) -> Tuple[Array, Array]:
        """
        _get_eigenstates Return eigenvalues and eigenvectors of the native Hamiltonian.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of (eigenvalues, eigenvectors).
        """
        hamiltonian = self._get_hamiltonian()
        eig_vals, eig_states = jsp.linalg.eigh(hamiltonian)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals, eig_states

    def add_charge_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_charge_drive Adds a charge drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        pulse : Pulse
            The pulse function of the drive.
        """
        drive = ChargeDrive(label, pulse)
        self._drives[label] = drive

    def add_flux_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_flux_drive Adds a flux drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        pulse : Pulse
            The pulse function of the drive.
        """
        drive = FluxDrive(label, pulse)
        self._drives[label] = drive

    def add_cosflux_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_cosflux_drive Adds a cos(flux) flux drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        pulse : Pulse
            The pulse function of the drive.
        """
        drive = CosFluxDrive(label, pulse)
        self._drives[label] = drive

    def add_sinflux_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_sinflux_drive Adds a sin(flux) flux drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        pulse : Pulse
            The pulse function of the drive.
        """
        drive = SinFluxDrive(label, pulse)
        self._drives[label] = drive


class ChargeTransmon(BaseTransmon):
    """Transmon qubit model."""

    def __init__(
        self,
        label: str,
        charging_energy: Float,
        josephson_energy: Float,
        offset_charge: Float,
        charge_cutoff: int,
        device_ind: int | None = None,
        device_dims: Iterable[int] | None = None,
    ) -> None:
        dim = int(2 * charge_cutoff + 1)
        super().__init__(
            label=label,
            charging_energy=charging_energy,
            josephson_energy=josephson_energy,
            offset_charge=offset_charge,
            charge_cutoff=charge_cutoff,
            dim=dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )

    def embed(self, device_ind: int, device_dims: Iterable[int]) -> ChargeTransmon:
        """
        embed Embeds the transmon into a larger Hilbert space.

        Parameters
        ----------
        device_ind : int
            The index of the transmon in the larger Hilbert space.
        device_dims : Tuple[int, ...]
            The dimension of each quantum system (including this transmon)
            in the full device.
        """
        transmon = self.__class__(
            label=self.label,
            charging_energy=self.charging_energy,
            josephson_energy=self.josephson_energy,
            offset_charge=self.offset_charge,
            charge_cutoff=self.charge_cutoff,
            device_ind=device_ind,
            device_dims=device_dims,
        )
        for label, drive in self._drives.items():
            transmon._drives[label] = drive

        return transmon

    def get_hamiltonian(self) -> Array:
        hamiltonian = self._get_hamiltonian()
        return self.embed_op(hamiltonian)

    def get_eigenvalues(self) -> Array:
        return self._get_eigenvalues()

    def get_eigenstates(self) -> Tuple[Array, Array]:
        return self._get_eigenstates()


class Transmon(BaseTransmon):
    """Transmon qubit model."""

    _eig_vals: Array
    _eig_states: Array

    def __init__(
        self,
        label: str,
        charging_energy: Float,
        josephson_energy: Float,
        offset_charge: Float,
        charge_cutoff: int,
        dim: int,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(
            label=label,
            charging_energy=charging_energy,
            josephson_energy=josephson_energy,
            offset_charge=offset_charge,
            charge_cutoff=charge_cutoff,
            dim=dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )

        eig_vals, eig_states = self._get_eigenstates()
        self._eig_vals = eig_vals[..., : self.dim]
        self._eig_states = eig_states[..., : self.dim]

    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Transmon:
        """
        embed Embeds the transmon into a larger Hilbert space.

        Parameters
        ----------
        device_ind : int
            The index of the transmon in the larger Hilbert space.
        device_dims : Tuple[int, ...]
            The dimension of each quantum system (including this transmon)
            in the full device.
        """
        transmon = self.__class__(
            label=self.label,
            charging_energy=self.charging_energy,
            josephson_energy=self.josephson_energy,
            offset_charge=self.offset_charge,
            charge_cutoff=self.charge_cutoff,
            dim=self.dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )

        for label, drive in self._drives.items():
            transmon._drives[label] = drive
        return transmon

    def transform_op(self, operator: Array) -> Array:
        """
        transform_op Transforms an operator of the energy basis of the transmon.

        Parameters
        ----------
        operator : Array
            The operator to process.
        Returns
        -------
        Array
            The processed operator.
        """
        transformed_op = transform_op(operator, self._eig_states)
        return transformed_op

    def process_op(self, operator: Array) -> Array:
        """
        process_op Process an operator into the transmon's current basis/embedding.

        Parameters
        ----------
        operator : Array
            Operator in the native basis.

        Returns
        -------
        Array
            Operator in the system's current representation.
        """
        return self.embed_op(self.transform_op(operator))

    def get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the fluxonium.

        Returns
        -------
        Array
            The number operator of the fluxonium.
        """
        num_op = jnp.diag(jnp.arange(self.dim))
        num_op = self.embed_op(num_op)
        return num_op

    def get_hamiltonian(self) -> Array:
        hamiltonian = jnp.diag(self._eig_vals)
        hamiltonian = self.embed_op(hamiltonian)
        return hamiltonian

    def get_eigenvalues(self) -> Array:
        return self._eig_vals

    def get_eigenstates(self) -> Tuple[Array, Array]:
        eig_states = jnp.identity(self.dim, dtype=jnp.complex128)
        return self._eig_vals, eig_states

    @classmethod
    def from_frequencies(
        cls,
        label: str,
        frequency: Float,
        anharmonicity: Float,
        offset_charge: Float,
        charge_cutoff: int,
        dim: int,
        *,
        atol: float = 1e-8,
        rtol: float = 1e-8,
    ) -> Transmon:
        """
        from_frequencies Create a Transmon based on the qubit frequency and anharmonicity. The function will optimize the charging and Josephson energies to match the provided frequency and anharmonicity. The optimization is done using the scipy.optimize.minimize function. The optimization is done in the following way:
        1. Calculate the initial guesses for the maximum Josephson energy and Charging energy based on the provided frequency and anharmonicity.
        2. Define an objective function that calculates the difference between the provided frequency and anharmonicity and the calculated frequency and anharmonicity based on the charging and Josephson energies. The objective function is the sum of the squared differences between the provided and calculated values.
        3. Optimize the objective function to find the charging and Josephson energies that best match the provided frequency and anharmonicity.

        Parameters
        ----------
        label : str
            The label of the transmon.
        frequency : ScalarLike
            The target qubit frequency.
        anharmonicity : Float
            The target qubit anharmonicity (should be negative).
        offset_charge : Float
            The offset charge of the transmon.
        charge_cutoff : int
            The number of charge states to consider.
        dim : int
            The dimension of the transmon Hilbert space.

        Returns
        -------
        Transmon
            The Transmon instance.
        """
        frequency = jnp.array(frequency)
        anharmonicity = jnp.array(anharmonicity)

        init_ec = -anharmonicity
        init_ej = (frequency + init_ec) ** 2 / (8 * init_ec)

        def objective_func(x, _) -> Array:
            charging_energy, josephson_energy = x
            transmon = Transmon(
                label,
                charging_energy,
                josephson_energy,
                offset_charge,
                charge_cutoff,
                dim,
            )
            freq_diff = frequency - transmon.frequency
            anharm_diff = anharmonicity - transmon.anharmonicity
            result = freq_diff**2 + anharm_diff**2
            return result

        init_guess = (init_ej, init_ec)
        solver = BFGS(rtol, atol)
        solution = minimise(objective_func, solver, init_guess)

        charging_energy, josephson_energy = solution.value
        transmon = cls(
            label=label,
            charging_energy=charging_energy,
            josephson_energy=josephson_energy,
            offset_charge=offset_charge,
            charge_cutoff=charge_cutoff,
            dim=dim,
        )
        return transmon
