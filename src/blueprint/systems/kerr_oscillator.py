import math
from typing import Callable, Dict, Self, Tuple

from equinox import field
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Scalar

from ..drives import BaseDrive as Drive
from ..util.linalg import cosm, embed_op, sinm
from .system import System

type Pulse = Callable[[float], Scalar | Array]


class KerrOscillator(System):
    """Kerr oscillator class for representing a weakly anharmonic oscillator."""

    label: str = field(static=True)
    _ec: Array
    _ej: Array
    dim: int = field(static=True)

    _drives: Dict[str, Drive]

    device_ind: int | None = field(static=True, default=None)
    device_dims: Tuple[int, ...] | None = field(static=True, default=None)

    def __init__(
        self,
        label: str,
        charging_energy: ArrayLike,
        josephson_energy: ArrayLike,
        dim: int,
        *,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ):
        self.label = str(label)
        self.dim = int(dim)
        self._ec = jnp.asarray(charging_energy)
        self._ej = jnp.asarray(josephson_energy)

        self._drives = {}

        self.device_ind = device_ind
        self.device_dims = device_dims

    def __check_init__(self) -> None:
        if not isinstance(self._ec, Array):
            raise TypeError("The charging_energy must be a jax.ArrayLike type.")

        if not isinstance(self._ej, Array):
            raise TypeError("The josephson_energy must be a jax.ArrayLike type.")

    @property
    def charging_energy(self) -> float:
        """
        charging_energy Returns the charging energy of the Kerr oscillator.

        Returns
        -------
        float
            The charging energy of the Kerr oscillator.
        """
        return self._ec

    @property
    def josephson_energy(self) -> float:
        """
        josephson_energy Returns the josephson energy of the Kerr oscillator.

        Returns
        -------
        float
            The josephson energy of the Kerr oscillator.
        """
        return self._ej

    @property
    def is_diagonal(self) -> bool:
        """
        is_diagonal Returns whether the Kerr oscillator is diagonalized.

        Returns
        -------
        bool
            True if the Kerr oscillator is diagonalized, False otherwise.
        """
        return True

    @property
    def plasma_frequency(self) -> float:
        """
        plasma_frequency Returns the plasma_frequency of the Kerr oscillator.

        Returns
        -------
        Array
            The plasma_frequency of the Kerr oscillator.
        """
        freq = jnp.sqrt(8 * self._ec * self._ej)
        return freq
    
    @property 
    def qubit_frequency(self) -> float:
        """
        qubit_frequency Returns the qubit frequency of the Kerr oscillator.

        Returns
        -------
        Array
            The qubit frequency of the Kerr oscillator.
        """
        freq = self.plasma_frequency - self._ec
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
        return (self._ej / (32 * self._ec)) ** 0.25

    @property
    def flux_zpf(self) -> float:
        """
        flux_zpf Returns the zero-point fluctuations of the flux.

        Returns
        -------
        float
            The zero-point fluctuations of the flux.
        """
        return (2 * self._ec / self._ej) ** 0.25
    
    def diagonalize(self) -> Self:
        return self # ! Not sure what this is for 

    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Self:
        """
        embed Embeds the Kerr oscillator into a larger Hilbert space.

        Parameters
        ----------
        ind : int
            The index of the Kerr oscillator in the larger Hilbert space.
        device_dims : Tuple[int]
            The dimension of each quantum system (including this Kerr oscillator) in the full device.
        """

        embedded_kerr_oscillator = KerrOscillator(
            label=self.label,
            charging_energy=self.charging_energy,
            josephson_energy=self.josephson_energy,
            dim=self.dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )

        return embedded_kerr_oscillator

    def process_op(self, operator: Array, embed: bool = True) -> Array:
        """
        process_op Processes an operator of the Kerr oscillator.

        Parameters
        ----------
        operator : Array
            The operator to process.
        diagonalize : bool, optional
            Whether to transform the operator to the energy eigenbasis of the Kerr oscillator, by default True
        embed : bool, optional
            Whether to embed the operator in a larger Hilbert space , by default True

        Returns
        -------
        Array
            The processed operator.
        """
        if embed and self.is_embedded:
            operator = embed_op(operator, self.device_ind, self.device_dims)

        return operator

    def _get_raise_op(self) -> Array:
        """
        _get_raise_op Returns the raising (creation) operator of the Kerr oscillator.

        Returns
        -------
        Array
            The raising (creation) operator of the Kerr oscillator.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self.dim))
        raise_op = jnp.diag(offdiag, k=-1)
        return raise_op

    def get_raise_op(self) -> Array:
        """
        get_creation_op Returns the raising (creation) operator of the Kerr oscillator.

        Returns
        -------
        Array
            The raising (creation) operator in the current basis of the Kerr oscillator.
        """
        raise_op = self._get_raise_op()
        return self.process_op(raise_op)

    def _get_low_op(self) -> Array:
        """
        _get_low_op Returns the lowering (annihilation) operator of the Kerr oscillator.

        Returns
        -------
        Array
            The lowering (annihilation) operator of the Kerr oscillator.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self.dim))
        low_op = jnp.diag(offdiag, k=1)
        return low_op

    def get_low_op(self) -> Array:
        """
        get_low_op Returns the lowering (annihilation) operator of the Kerr oscillator.

        Returns
        -------
        Array
            The lowering (annihilation) operator in the current basis of the Kerr oscillator.
        """
        low_op = self._get_low_op()
        return self.process_op(low_op)

    def _get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the Kerr oscillator.

        Returns
        -------
        Array
            The number operator of the Kerr oscillator.
        """
        diag_elems = jnp.arange(self.dim)
        number_op = jnp.diag(diag_elems)
        return number_op

    def get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the Kerr oscillator.

        Returns
        -------
        Array
            The number operator in the current basis of the Kerr oscillator.
        """
        number_op = self._get_number_op()
        return self.process_op(number_op)

    def _get_identity_op(self) -> Array:
        """
        get_identity_op Returns the identity operator of the Kerr oscillator.

        Returns
        -------
        Array
            The identity operator of the Kerr oscillator.
        """
        id_op = jnp.identity(self.dim)
        return id_op

    def get_identity_op(self) -> Array:
        """
        get_identity_op Returns the identity operator of the Kerr oscillator.

        Returns
        -------
        Array
            The identity operator of the Kerr oscillator.
        """
        id_op = self._get_identity_op()
        return self.process_op(id_op)

    def _get_charge_op(self) -> Array:
        """
        _get_charge_op Returns the charge operator of the Kerr oscillator in the Fock basis.

        Returns
        -------
        Array
            The charge operator of the Kerr oscillator, expressed in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        charge_op = 1.0j * self.charge_zpf * (raise_op - low_op)
        return charge_op

    def get_charge_op(self) -> Array:
        """
        get_charge_op Returns the charge operator of the Kerr oscillator.

        Returns
        -------
        Array
            The charge operator, in the current basis of the Kerr oscillator.
        """
        charge_op = self._get_charge_op()
        processed_op = self.process_op(charge_op)
        return processed_op

    def _get_flux_op(self) -> Array:
        """
        _get_flux_op Returns the flux operator of the Kerr oscillator in the Fock basis.

        Returns
        -------
        Array
            The flux operator of the Kerr oscillator, expressed in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        flux_op = self.flux_zpf * (raise_op + low_op)
        return flux_op

    def get_flux_op(self) -> Array:
        """
        get_flux_op Returns the flux operator of the Kerr oscillator.

        Returns
        -------
        Array
            The flux operator, in the current basis of the Kerr oscillator.
        """
        charge_op = self._get_flux_op()
        processed_op = self.process_op(charge_op)
        return processed_op

    def _get_self_kerr_op(self) -> Array:
        """
        _get_self_kerr_op Returns the self-Kerr operator of the Kerr oscillator.

        Returns
        -------
        Array
            The self-Kerr operator of the Kerr oscillator.
        """

        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        self_kerr_op = raise_op @ raise_op @ low_op @ low_op
        return self_kerr_op

    def get_self_kerr_op(self) -> Array:
        """
        get_self_kerr_op Returns the self-Kerr operator of the Kerr oscillator.

        Returns
        -------
        Array
            The self-Kerr operator in the current basis of the Kerr oscillator.
        """
        self_kerr_op = self._get_self_kerr_op()
        processed_op = self.process_op(self_kerr_op)
        return processed_op

    def _get_cosphi_op(self) -> Array:
        """
        _get_cosphi_op Returns the cos(phi) operator of the Kerr oscillator in the Fock basis.

        Returns
        -------
        Array
            The cos(phi) operator of the Kerr oscillator, expressed in the Fock basis.
        """
        flux_op = self._get_flux_op()
        cosphi_op = cosm(flux_op)
        return cosphi_op

    def get_cosphi_op(self) -> Array:
        """
        get_cosphi_op Returns the cos(phi) operator of the Kerr oscillator.

        Returns
        -------
        Array
            The cos(phi) operator, in the current basis of the Kerr oscillator.
        """
        cosphi_op = self._get_cosphi_op()
        processed_op = self.process_op(cosphi_op)
        return processed_op

    def _get_sinphi_op(self) -> Array:
        """
        _get_sinphi_op Returns the sin(phi) operator of the Kerr oscillator in the Fock basis.

        Returns
        -------
        Array
            The sin(phi) operator of the Kerr oscillator, expressed in the Fock basis.
        """
        flux_op = self._get_flux_op()
        sinphi_op = sinm(flux_op)
        return sinphi_op

    def get_sinphi_op(self) -> Array:
        """
        get_sinphi_op Returns the sin(phi) operator of the Kerr oscillator.

        Returns
        -------
        Array
            The sin(phi) operator, in the current basis of the Kerr oscillator.
        """
        sinphi_op = self._get_sinphi_op()
        processed_op = self.process_op(sinphi_op)
        return processed_op

    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Returns the Hamiltonian of the Kerr oscillator.

        Returns
        -------
        Array
            The Hamiltonian of the Kerr oscillator.
        """
        number_op = self._get_number_op()
        self_kerr_op = self._get_self_kerr_op()
        hamiltonian = self.plasma_frequency * number_op - (self._ec / 2) * self_kerr_op
        return hamiltonian

    def get_hamiltonian(self) -> Array:
        hamiltonian = self._get_hamiltonian()
        return self.process_op(hamiltonian)

    def get_eigenvalues(self) -> Array:
        excitation_number = jnp.arange(self.dim)
        eig_vals = (
            excitation_number * (self.qubit_frequency + self._ec / 2)
            - (self._ec / 2) * excitation_number**2
        )
        return eig_vals

    def get_eigenstates(self) -> Array:
        eig_states = jnp.identity(self.dim, dtype=complex)
        excitation_number = jnp.arange(self.dim)
        eig_vals = (
            excitation_number * (self.qubit_frequency + self._ec / 2)
            - (self._ec / 2) * excitation_number**2
        )
        return eig_vals, eig_states

    @classmethod
    def from_frequency(
        cls,
        label: str,
        frequency: float,
        anharmonicity: float,
        dim: int,
        *,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> Self:
        """
        from_frequency Returns a Kerr oscillator object from the frequency and the anharmonicity.

        Parameters
        ----------
        label : str
            The label of the Kerr oscillator.
        frequency : float
            The frequency of the Kerr oscillator.
        anharmonicity : float
            The anharmonicity of the Kerr oscillator.
        dim : int
            The dimension of the Hilbert space for the Kerr oscillator.


        Returns
        -------
        KerrOscillator
            The Kerr oscillator object.
        """
        charging_energy = anharmonicity
        josephson_energy = (frequency + charging_energy) ** 2 / (8 * charging_energy)

        KerrOscillator = cls(
            label=label,
            charging_energy=charging_energy,
            josephson_energy=josephson_energy,
            dim=dim,
            device_ind=device_ind,
            device_dims=device_dims,
        )
        return KerrOscillator
