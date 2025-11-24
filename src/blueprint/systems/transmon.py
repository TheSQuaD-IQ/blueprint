from abc import abstractmethod
from typing import Callable, Tuple, Self

from jax import numpy as jnp
from jax import scipy as jsp
from jaxtyping import Scalar, ArrayLike, Array

from equinox import field

from .system import System
from ..drives import ChargeDrive, FluxDrive, CosFluxDrive, SinFluxDrive
from ..util.linalg import transform_op, embed_op

type Pulse = Callable[[float], Scalar | Array]


class BaseTransmon(System):
    """Transmon qubit model."""

    _ec: Array
    _ej: Array
    _ng: Array
    _ncut: int = field(static=True)

    @property
    def charging_energy(self) -> float:
        """
        charging_energy Returns the charging energy of the transmon.

        Returns
        -------
        float
            The charging energy of the transmon.
        """
        return self._ec

    @property
    def josephson_energy(self) -> float:
        """
        josephson_energy Returns the Josephson energy of the transmon.

        Returns
        -------
        float
            The Josephson energy of the transmon.
        """
        return self._ej

    @property
    def offset_charge(self) -> float:
        """
        offset_charge Returns the offset charge of the transmon

        Returns
        -------
        float
            The offset charge of the transmon.
        """
        return self._ng

    @property
    def charge_cutoff(self) -> int:
        """
        charge_cutoff Returns the number of charge states to consider.

        Returns
        -------
        int
            The number of charge states to consider.
        """
        return self._ncut

    @property
    @abstractmethod
    def is_diagonal(self) -> bool:
        """
        is_diagonal Returns whether the transmon is diagonalized.

        Returns
        -------
        bool
            Whether the transmon is diagonalized.
        """

    @property
    def approx_frequency(self) -> Array:
        """
        approx_frequency Returns the approximate 0-1 frequency of the transmon.

        Returns
        -------
        float
            The approximate transmon 0-1 frequency.
        """
        sqrt_term = jnp.sqrt(8 * self._ec * self._ej)
        return sqrt_term - self._ec

    @property
    def charge_zpf(self) -> float:
        """
        charge_zpf Returns the zero-point fluctuations of the charge variable of the transmon.
        Note that this is only defined in the energy eigenbasis of the transmon, meaning
        that the transmon must be diagonalized before calling this property.

        Returns
        -------
        float
            The zero-point fluctuations of the charge variable.

        Raises
        ------
        ValueError
            If the transmon is not diagonalized.
        """
        charge_fluctuations = (self._ej / (32 * self._ec)) ** 0.25

        # charge_vals = jnp.arange(-self._ncut, self._ncut + 1)
        # charge_op = jnp.diag(charge_vals)
        # diag_op = transform_op(charge_op, self._eig_states)
        # squared_op = diag_op @ diag_op
        # exp_val = squared_op[0, 0]
        # charge_fluctuations = math.sqrt(float(exp_val.real))

        return charge_fluctuations

    @property
    def flux_zpf(self) -> float:
        """
        charge_zpf Returns the zero-point fluctuations of the flux variable of the transmon.
        Note that this is only defined in the energy eigenbasis of the transmon, meaning
        that the transmon must be diagonalized before calling this property.

        Returns
        -------
        float
            The zero-point fluctuations of the flux variable.

        Raises
        ------
        ValueError
            If the transmon is not diagonalized.
        """
        flux_fluctuations = (2 * self._ec / self._ej) ** 0.25
        # flux_fluctuations =  1 / (2 * self.charge_zpf)
        return flux_fluctuations

    @abstractmethod
    def process_op(self, operator: Array) -> Array:
        """
        process_op Processes an operator of the transmon.

        Parameters
        ----------
        operator : Array
            The operator to process.

        Returns
        -------
        Array
            The processed operator.
        """

    def _get_charge_op(self) -> Array:
        """
        _get_charge_op Returns the charge operator of the transmon in the charge basis.

        Parameters
        ----------
        include_charge_offset : bool
            Whether to include the charge offset

        Returns
        -------
        Array
            The charge operator of the transmon expressed in the charge basis.
        """
        charge_vals = jnp.arange(-self._ncut, self._ncut + 1)
        charge_op = jnp.diag(charge_vals)

        id_op = self._get_identity_op()
        offset_op = self._ng * id_op

        offset_charge_op = charge_op - offset_op
        return offset_charge_op

    def get_charge_op(self) -> Array:
        """
        get_charge_op Returns the charge operator of the fluxonium.

        Returns
        -------
        Array
            The charge operator, in the current basis of the fluxonium.
        """
        native_op = self._get_charge_op()
        return self.process_op(native_op)

    def _get_cosphi_op(self) -> Array:
        """
        _get_cosphi_op Returns the cos(phi) operator of the transmon in the charge basis.

        Returns
        -------
        Array
            The cos(phi) operator of the transmon expressed in the charge basis.
        """
        offdiag_elems = jnp.ones(2 * self._ncut, dtype=jnp.complex64)
        superdiag_mat = jnp.diag(0.5 * offdiag_elems, 1)
        subdiag_mat = jnp.transpose(superdiag_mat)
        op = superdiag_mat + subdiag_mat
        return op

    def get_cosphi_op(self) -> Array:
        """
        cosphi_op Returns the cos(phi) operator of the transmon.

        Returns
        -------
        Array
            The cos(phi) operator in the current basis of the transmon.
        """
        native_op = self._get_cosphi_op()
        op = self.process_op(native_op)
        return op

    def _get_sinphi_op(self) -> Array:
        """
        _get_cosphi_op Returns the cos(phi) operator of the transmon in the charge basis.

        Returns
        -------
        Array
            The cos(phi) operator of the transmon expressed in the charge basis.
        """
        offdiag_elems = jnp.ones(2 * self._ncut)
        superdiag_mat = jnp.diag(0.5j * offdiag_elems, 1)
        subdiag_mat = jnp.transpose(superdiag_mat)
        op = superdiag_mat - subdiag_mat
        return op

    def get_sinphi_op(self) -> Array:
        """
        cosphi_op Returns the cos(phi) operator of the transmon.

        Returns
        -------
        Array
            The cos(phi) operator in the current basis of the transmon.
        """
        native_op = self._get_sinphi_op()
        op = self.process_op(native_op)
        return op

    def _get_identity_op(self) -> Array:
        """
        _get_identity_op Returns the identity operator of the transmon in the charge basis.

        Returns
        -------
        Array
            The identity operator of the transmon expressed in the charge basis.
        """
        charge_dim = 2 * self._ncut + 1
        id_op = jnp.identity(charge_dim)
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
        processed_op = self.process_op(id_op)
        return processed_op

    def _get_kinetic_term(self) -> Array:
        """
        _get_kinetic_term Returns the kinetic term of the Hamiltonian in the charge basis.

        Returns
        -------
        Array
            The kinetic term of the transmon Hamiltonian expressed in the charge basis.
        """
        offset_charge_op = self._get_charge_op()
        kinetic_term = 4 * self._ec * offset_charge_op @ offset_charge_op
        return kinetic_term

    def _get_potential_term(self) -> Array:
        """
        _get_potential_term Returns the potential term of the Hamiltonian in the charge basis.

        Returns
        -------
        Array
            The potential term of the transmon Hamiltonian expressed in the charge basis.
        """
        cosphi_op = self._get_cosphi_op()

        potential_term = -self._ej * cosphi_op
        return potential_term

    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Returns the Hamiltonian of the transmon in the charge basis.

        Returns
        -------
        Array
            The Hamiltonian of the transmon expressed in the charge basis.
        """
        kinetic_term = self._get_kinetic_term()
        potential_term = self._get_potential_term()
        hamiltonian = kinetic_term + potential_term
        return hamiltonian

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
        drive = ChargeDrive(label=label, pulse=pulse)
        self.drives[label] = drive

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
        drive = FluxDrive(label=label, pulse=pulse)
        self.drives[label] = drive

    def add_cosflux_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_cosflux_drive Adds a cos(phi) flux drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        pulse : Pulse
            The pulse function of the drive.
        """
        drive = CosFluxDrive(label=label, pulse=pulse)
        self.drives[label] = drive

    def add_sinflux_drive(self, label: str, pulse: Pulse) -> None:
        """
        add_sinflux_drive Adds a sin(phi) flux drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        pulse : Pulse
            The pulse function of the drive.
        """
        drive = SinFluxDrive(label=label, pulse=pulse)
        self.drives[label] = drive


class ChargeTransmon(BaseTransmon):
    """Transmon qubit model."""

    def __init__(
        self,
        label: str,
        charging_energy: ArrayLike,
        josephson_energy: ArrayLike,
        offset_charge: ArrayLike,
        charge_cutoff: int,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> None:
        self.label = str(label)

        self._ec = jnp.asarray(charging_energy)
        self._ej = jnp.asarray(josephson_energy)
        self._ng = jnp.asarray(offset_charge)

        self._ncut = int(charge_cutoff)
        self.dim = 2 * charge_cutoff + 1

        self.drives = {}

        self.device_ind = device_ind
        self.device_dims = device_dims

    @property
    def is_diagonal(self) -> bool:
        """
        is_diagonal Returns whether the transmon is diagonalized.

        Returns
        -------
        bool
            Whether the transmon is diagonalized.
        """
        return False

    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Self:
        """
        embed Embeds the transmon into a larger Hilbert space.

        Parameters
        ----------
        ind : int
            The index of the transmon in the larger Hilbert space.
        device_dims : Tuple[int]
            The dimension of each quantum system (including this transmon)
            in the full device.
        """
        transmon = ChargeTransmon(
            label=self.label,
            charging_energy=self.charging_energy,
            josephson_energy=self.josephson_energy,
            offset_charge=self.offset_charge,
            charge_cutoff=self.charge_cutoff,
            device_ind=device_ind,
            device_dims=device_dims,
        )
        for label, drive in self.drives.items():
            transmon.drives[label] = drive
        return transmon

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

    def get_hamiltonian(self) -> Array:
        hamiltonian = self._get_hamiltonian()
        return self.process_op(hamiltonian)

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
        charging_energy: ArrayLike,
        josephson_energy: ArrayLike,
        offset_charge: ArrayLike,
        charge_cutoff: int,
        dim: int,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> None:
        self.label = str(label)

        self._ec = jnp.asarray(charging_energy)
        self._ej = jnp.asarray(josephson_energy)
        self._ng = jnp.asarray(offset_charge)

        self._ncut = int(charge_cutoff)
        self.dim = int(dim)

        self.drives = {}

        self.device_ind = device_ind
        self.device_dims = device_dims

        eig_vals, eig_states = self._get_eigenstates()
        self._eig_vals = eig_vals[..., : self.dim]
        self._eig_states = eig_states[..., : self.dim]

    @property
    def is_diagonal(self) -> bool:
        """
        is_diagonal Returns whether the transmon is diagonalized.

        Returns
        -------
        bool
            Whether the transmon is diagonalized.
        """
        return True

    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Self:
        """
        embed Embeds the transmon into a larger Hilbert space.

        Parameters
        ----------
        ind : int
            The index of the transmon in the larger Hilbert space.
        device_dims : Tuple[int]
            The dimension of each quantum system (including this transmon)
            in the full device.
        """
        transmon = Transmon(
            label=self.label,
            charging_energy=self.charging_energy,
            josephson_energy=self.josephson_energy,
            offset_charge=self.offset_charge,
            charge_cutoff=self.charge_cutoff,
            dim=self.dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )

        for label, drive in self.drives.items():
            transmon.drives[label] = drive
        return transmon

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

        if self.is_embedded:
            return embed_op(transformed_op, self.device_ind, self.device_dims)

        return transformed_op

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

    def get_hamiltonian(self) -> Array:
        hamiltonian = jnp.diag(self._eig_vals)

        if self.is_embedded:
            return embed_op(hamiltonian, self.device_ind, self.device_dims)

        return hamiltonian

    def get_eigenvalues(self) -> Array:
        return self._eig_vals

    def get_eigenstates(self) -> Array:
        eig_states = jnp.identity(self.dim, dtype=jnp.complex128)
        return self._eig_vals, eig_states
