from typing import Callable, Self, Tuple

from jax import numpy as jnp
from jaxtyping import Array, Scalar

from ..util.linalg import cosm, sinm
from .system import System

type Pulse = Callable[[Scalar], Array]


class KerrOscillator(System):
    """Kerr oscillator class for representing a weakly anharmonic oscillator."""

    _ec: Scalar
    _ej: Scalar
    _kerr_sign: Scalar

    def __init__(
        self,
        label: str,
        charging_energy: float | Scalar,
        josephson_energy: float | Scalar,
        dim: int,
        kerr_sign: float | Scalar = -1.0,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ):
        super().__init__(label, dim, device_ind=device_ind, device_dims=device_dims)

        self._ec = jnp.asarray(charging_energy)
        self._ej = jnp.asarray(josephson_energy)
        self._kerr_sign = jnp.asarray(kerr_sign)

    @property
    def kerr_sign(self) -> Scalar:
        """
        kerr_sign Returns the sign of the self-Kerr nonlinearity.

        Returns
        -------
        Scalar
            The sign of the self-Kerr nonlinearity.
        """
        return self._kerr_sign

    @property
    def charging_energy(self) -> Scalar:
        """
        charging_energy Returns the charging energy of the Kerr oscillator.

        Returns
        -------
        Scalar
            The charging energy of the Kerr oscillator.
        """
        return self._ec

    @property
    def josephson_energy(self) -> Scalar:
        """
        josephson_energy Returns the josephson energy of the Kerr oscillator.

        Returns
        -------
        Scalar
            The josephson energy of the Kerr oscillator.
        """
        return self._ej

    @property
    def plasma_frequency(self) -> Scalar:
        """
        plasma_frequency Returns the plasma_frequency of the Kerr oscillator.

        Returns
        -------
        Scalar
            The plasma_frequency of the Kerr oscillator.
        """
        freq = jnp.sqrt(8 * self._ec * self._ej)
        return freq

    @property
    def charge_zpf(self) -> Scalar:
        """
        charge_zpf Returns the zero-point fluctuations of the charge.

        Returns
        -------
        Scalar
            The zero-point fluctuations of the charge.
        """
        return (self._ej / (32 * self._ec)) ** 0.25

    @property
    def flux_zpf(self) -> Scalar:
        """
        flux_zpf Returns the zero-point fluctuations of the flux.

        Returns
        -------
        Scalar
            The zero-point fluctuations of the flux.
        """
        return (2 * self._ec / self._ej) ** 0.25

    @property
    def _self_kerr(self) -> Scalar:
        """
        _self_kerr Returns the self-Kerr nonlinearity of the Kerr oscillator.

        Returns
        -------
        Scalar
            The self-Kerr nonlinearity of the Kerr oscillator.
        """
        return self._kerr_sign * self._ec

    def embed(self, device_ind: int, device_dims: Tuple[int, ...]) -> Self:
        """
        embed Embeds the Kerr oscillator into a larger Hilbert space.

        Parameters
        ----------
        device_ind : int
            Index of this system within the device.
        device_dims : tuple
            Device subsystem dimensions.
        """

        embedded_kerr_oscillator = self.__class__(
            self.label,
            self.charging_energy,
            self.josephson_energy,
            self.dim,
            self.kerr_sign,
            device_ind,
            device_dims,
        )

        for label, drive in self.drives.items():
            embedded_kerr_oscillator.drives[label] = drive

        return embedded_kerr_oscillator

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
        processed_op = self.embed_op(operator)
        return processed_op

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
        id_op = jnp.identity(self.dim)
        return self.embed_op(id_op)

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

    def _get_cosflux_op(self) -> Array:
        """
        _get_cosflux_op Returns the cos(phi) operator of the Kerr oscillator in the Fock basis.

        Returns
        -------
        Array
            The cos(phi) operator of the Kerr oscillator, expressed in the Fock basis.
        """
        flux_op = self._get_flux_op()
        cosflux_op = cosm(flux_op)
        return cosflux_op

    def get_cosflux_op(self) -> Array:
        """
        get_cosflux_op Returns the cos(phi) operator of the Kerr oscillator.

        Returns
        -------
        Array
            The cos(phi) operator, in the current basis of the Kerr oscillator.
        """
        cosflux_op = self._get_cosflux_op()
        processed_op = self.process_op(cosflux_op)
        return processed_op

    def _get_sinflux_op(self) -> Array:
        """
        _get_sinflux_op Returns the sin(phi) operator of the Kerr oscillator in the Fock basis.

        Returns
        -------
        Array
            The sin(phi) operator of the Kerr oscillator, expressed in the Fock basis.
        """
        flux_op = self._get_flux_op()
        sinflux_op = sinm(flux_op)
        return sinflux_op

    def get_sinflux_op(self) -> Array:
        """
        get_sinflux_op Returns the sin(phi) operator of the Kerr oscillator.

        Returns
        -------
        Array
            The sin(phi) operator, in the current basis of the Kerr oscillator.
        """
        sinflux_op = self._get_sinflux_op()
        processed_op = self.process_op(sinflux_op)
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
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        anharm_op = raise_op @ raise_op @ low_op @ low_op
        qubit_freq = self.plasma_frequency + self._self_kerr

        hamiltonian = qubit_freq * number_op + (0.5 * self._self_kerr) * anharm_op
        return hamiltonian

    def get_hamiltonian(self) -> Array:
        hamiltonian = self._get_hamiltonian()
        return self.process_op(hamiltonian)

    def get_eigenvalues(self) -> Array:
        exc_nums = jnp.arange(self.dim)
        harm_vals = exc_nums * self.plasma_frequency
        anharm_vals = 0.5 * self._self_kerr * exc_nums * (1 + exc_nums)
        eig_vals = harm_vals + anharm_vals
        return eig_vals

    def get_eigenstates(self) -> Tuple[Array, Array]:
        eig_vals = self.get_eigenvalues()
        eig_states = jnp.identity(self.dim, dtype=complex)
        return eig_vals, eig_states

    @classmethod
    def from_frequencies(
        cls,
        label: str,
        frequency: float | Scalar,
        anharmonicity: float | Scalar,
        dim: int,
        *,
        device_ind: int | None = None,
        device_dims: Tuple[int, ...] | None = None,
    ) -> Self:
        """
        from_frequencies Returns a Kerr oscillator object from the frequency and the anharmonicity.

        Parameters
        ----------
        label : str
            The label of the Kerr oscillator.
        frequency : float | Scalar
            The frequency of the Kerr oscillator.
        anharmonicity : float | Scalar
            The anharmonicity of the Kerr oscillator.
        dim : int
            The dimension of the Hilbert space for the Kerr oscillator.


        Returns
        -------
        KerrOscillator
            The Kerr oscillator object.
        """
        frequency = jnp.asarray(frequency)
        anharmonicity = jnp.asarray(anharmonicity)

        charging_energy = jnp.abs(anharmonicity)
        josephson_energy = (frequency - anharmonicity) ** 2 / (8 * charging_energy)
        kerr_sign = jnp.sign(anharmonicity)

        KerrOscillator = cls(
            label=label,
            charging_energy=charging_energy,
            josephson_energy=josephson_energy,
            dim=dim,
            kerr_sign=kerr_sign,
            device_ind=device_ind,
            device_dims=device_dims,
        )
        return KerrOscillator
