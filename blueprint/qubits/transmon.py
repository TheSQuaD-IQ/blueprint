import math
from typing import Union
from warnings import warn

from jax import numpy as jnp
from jax import Array

from ..base.systems import QuantumSystem

# NOTE: Separate class for fixed-frequency transmon?

class TunableTransmon(QuantumSystem):
    """
    Transmon qubit model.

    Args:
        charging_energy: The Charge energy of the transmon.
        josephson_energy: The maximum Joseph energy of the transmon.
        offset_charge: The gate charge of the transmon.
        ext_flux: The external flux trough the SQUID loop of the transmon.
        asymmetry: The asymmetry of the SQUID junctions.
        charge_cutoff: The number of charge states to consider.
        relax_time: The relaxation time of the transmon.
        deph_time: The dephasing time of the transmon.
        Note that this isn't the pure dephasing, i.e., it includes the relaxation time contribution.
    """

    def __init__(
        self,
        label: str,
        charging_energy: float,
        josephson_energy: float,
        offset_charge: float = 0.0,
        ext_flux: float = 0.0,
        asymmetry: float = 0.0,
        charge_cutoff: int = 100,
        relax_time: float | None = None,
        deph_time: float | None = None,
    ) -> None:

        # The transmon parameters (Josephson energy, charging energy, and gate charge)
        if not isinstance(charging_energy, float):
            raise ValueError(
                f"The charging energy expected to be a float, instead got type {type(charging_energy)}."
            )
        if josephson_energy < 0:
            raise ValueError("The Josephson energy must be a positive.")
        self._ec = charging_energy

        if not isinstance(josephson_energy, float):
            raise ValueError(
                f"The maximum Josephson energy expected to be a float, instead got type {type(josephson_energy)}."
            )
        if josephson_energy < 0:
            raise ValueError("The maximum Josephson energy must be a positive.")
        self._ej = josephson_energy

        if not isinstance(offset_charge, float):
            raise ValueError(
                f"The offset charge expected to be a float, instead got type {type(offset_charge)}."
            )
        self._ng = offset_charge

        if not isinstance(ext_flux, float):
            raise ValueError(
                f"The external flux expected to be a float, instead got type {type(ext_flux)}."
            )
        self._ext_flux = ext_flux

        if not isinstance(asymmetry, float):
            raise ValueError(
                f"The asymmetry expected to be a float, instead got type {type(asymmetry)}."
            )
        if abs(asymmetry) > 1:
            raise ValueError(
                "The absolute value of the asymmetry must be less than or equal to one."
            )
        self._asymm = asymmetry

        # The number of charge states to consider
        # when constructing the Hamiltonian/operators
        # in the native (charge) basis
        if not isinstance(charge_cutoff, int):
            raise ValueError(
                f"The charge cutoff expected to be an integer, "
                f"instead got type {type(charge_cutoff)}."
            )
        if charge_cutoff <= 0:
            raise ValueError(
                "The charge cutoff must be a non-negative integer or equal to zero."
            )
        self._ncut = charge_cutoff

        # The dimension of the Hilbert space
        dim = 2 * self._ncut + 1
        super().__init__(label, dim)

        # The relaxation and dephasing times
        if relax_time is not None:
            if not isinstance(relax_time, float):
                raise ValueError(
                    f"The relaxation time expected to be a float, instead got type {type(relax_time)}."
                )
            if relax_time < 0:
                raise ValueError("The relaxation time must be a non-negative float.")
        self._relax_time = relax_time

        if deph_time is not None:
            if not isinstance(deph_time, float):
                raise ValueError(
                    f"The dephasing time expected to be a float, instead got type {type(deph_time)}."
                )
            if deph_time < 0:
                raise ValueError("The dephasing time must be a non-negative float.")
            if relax_time is not None:
                if deph_time > 2 * relax_time:
                    raise ValueError(
                        "The dephasing time must be less than or equal to two times the relaxation time."
                    )
        self._deph_time = deph_time

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
    def max_josephson_energy(self) -> float:
        """
        josephson_energy Returns the Josephson energy of the transmon.

        Returns
        -------
        float
            The Josephson energy of the transmon.
        """
        return self._ej

    @property
    def _ej_eff(self) -> float:
        """
        eff_josephson_energy Returns the effective Josephson energy of the transmon.
        This is the Josephson energy modified by the external flux and the junction asymmetry.

        Returns
        -------
        float
            The effective Josephson energy
        """
        cos_term = math.cos(self._ext_flux)
        sqrt_term = math.sqrt(1 + self._asymm**2 * math.tan(self._ext_flux) ** 2)

        prefactor = abs(cos_term) * sqrt_term
        return self._ej * prefactor

    @property
    def eff_josephson_energy(self) -> float:
        """
        eff_josephson_energy Returns the effective Josephson energy of the transmon. This is the Josephson energy modified by the external flux and the junction asymmetry.

        Returns
        -------
        float
            The effective Josephson energy.
        """
        return self._ej_eff

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
    def external_flux(self) -> float:
        """
        external_flux Returns the external flux of the transmon.

        Returns
        -------
        float
            The external flux of the transmon.
        """
        return self._ext_flux

    @property
    def asymmetry(self) -> float:
        """
        asymmetry Returns the asymmetry of the SQUID junctions.

        Returns
        -------
        float
            The asymmetry of the SQUID junctions.
        """
        return self._asymm

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
    def flux_zpf(self) -> float:
        """
        flux_zpf Returns the zero-point fluctuations of the flux variable of the transmon.

        Returns
        -------
        float
            The zero-point fluctuations of the flux variable.
        """
        return (2 * self._ec / self._ej_eff) ** 0.25

    @property
    def charge_zpf(self) -> float:
        """
        charge_zpf Returns the zero-point fluctuations of the charge variable of the transmon.

        Returns
        -------
        float
            The zero-point fluctuations of the charge variable.
        """
        return (self._ej_eff / (32 * self._ec)) ** 0.25

    def _get_charge_op(self) -> Array:
        """
        _get_charge_op Returns the charge operator of the transmon in the charge basis.

        Returns
        -------
        Array
            The charge operator of the transmon, expressed in the charge basis.
        """
        charge_vals = jnp.arange(-self._ncut, self._ncut + 1)
        op = jnp.diag(charge_vals)
        return op

    def get_charge_op(self) -> Array:
        """
        charge_op Returns the charge operator of the transmon.

        Returns
        -------
        Array
            The charge operator, in the current basis of the transmon.
        """
        native_op = self._get_charge_op()
        op = self.process_op(native_op)
        return op

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
        offdiag_elems = jnp.ones(2 * self._ncut, dtype=jnp.complex64)
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
    

    def _get_kinetic_term(self) -> Array:
        """
        _get_kinetic_term Returns the kinetic term of the Hamiltonian in the charge basis.

        Returns
        -------
        Array
            The kinetic term of the transmon Hamiltonian expressed in the charge basis.
        """
        charge_dim = 2 * self._ncut + 1

        n_op = self._get_charge_op()
        id_op = jnp.identity(charge_dim)

        n_offset_op = n_op - self._ng * id_op
        kinetic_term = 4 * self._ec * n_offset_op @ n_offset_op
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
        sinphi_op = self._get_cosphi_op()

        phase = jnp.arctan(self._asymm * jnp.tan(self._ext_flux))

        cos_term = cosphi_op * jnp.cos(phase)
        sin_term = sinphi_op * jnp.sin(phase)

        potential_term = -self._ej_eff * (cos_term + sin_term)
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
        hamil = kinetic_term + potential_term

        return hamil

    def get_potential(self, phases: Union[float, Array]) -> Array:
        """
        potential Returns the potential energy of the transmon.

        Parameters
        ----------
        phases : Array
            The phase values.

        Returns
        -------
        Array
            The potential energy.
        """
        potential = -self._ej_eff * jnp.cos(phases)
        return potential


class AnharmonicOscillator(QuantumSystem):
    """
    AnharmonicOscillator An approximate trasmon model as an anharmonic oscillator.

    """
    def __init__(
        self,
        label: str,
        frequency: float,
        anharmonicity: float,
        ext_flux: float = 0.0,
        dim: int = 2,
        relax_time: float | None = None,
        deph_time: float | None = None,
    ) -> None:
        if not isinstance(frequency, float):
            raise ValueError(
                f"The maximum frequency expected to be a float, instead got type {type(frequency)}."
            )
        if frequency < 0:
            raise ValueError("The frequency must be a positive.")
        self._freq = frequency

        if not isinstance(anharmonicity, float):
            raise ValueError(
                f"The anharmonicity expected to be a float, instead got type {type(anharmonicity)}."
            )
        if anharmonicity > 0:
            warn("The anharmonicity is typically negative for transmon qubits. Instead a positive anharmonicity was provided.")
        self._anharm = anharmonicity

        if not isinstance(ext_flux, float):
            raise ValueError(
                f"The external flux expected to be a float, instead got type {type(ext_flux)}."
            )
        self._ext_flux = ext_flux

        super().__init__(label, dim)

        self._diagonalized = True

        # The relaxation and dephasing times
        if relax_time is not None:
            if not isinstance(relax_time, float):
                raise ValueError(
                    f"The relaxation time expected to be a float, instead got type {type(relax_time)}."
                )
            if relax_time < 0:
                raise ValueError("The relaxation time must be a non-negative float.")
        self._relax_time = relax_time

        if deph_time is not None:
            if not isinstance(deph_time, float):
                raise ValueError(
                    f"The dephasing time expected to be a float, instead got type {type(deph_time)}."
                )
            if deph_time < 0:
                raise ValueError("The dephasing time must be a non-negative float.")
            if relax_time is not None:
                if deph_time > 2 * relax_time:
                    raise ValueError(
                        "The dephasing time must be less than or equal to two times the relaxation time."
                    )
        self._deph_time = deph_time

    @property
    def frequency(self) -> float:
        """
        frequency Returns the frequency of the transmon.

        Returns
        -------
        float
            The frequency of the transmon.
        """
        res_freq = self._freq - self._anharm
        cos_term = math.cos(math.pi * self._ext_flux)
        shifted_freq = res_freq * math.sqrt(abs(cos_term))
        return shifted_freq + self._anharm
    
    @property
    def max_frequency(self) -> float:
        """
        max_frequency Returns the maximum frequency of the transmon.

        Returns
        -------
        float
            The maximum frequency of the transmon.
        """
        return self._freq
    
    @property
    def anharmonicity(self) -> float:
        """
        anharmonicity Returns the anharmonicity of the transmon.

        Returns
        -------
        float
            The anharmonicity of the transmon.
        """
        return self._anharm
    
    @property
    def ext_flux(self) -> float:
        """
        ext_flux Returns the external flux through the SQUID loop of the transmon.

        Returns
        -------
        float
            The external flux.
        """
        return self._ext_flux
    
    @property
    def charge_energy(self) -> float:
        """
        charge_energy Returns the charging energy of the transmon.

        Returns
        -------
        float
            The charging energy of the transmon.
        """
        return -self._anharm

    @property
    def josephson_energy(self) -> float:
        """
        josephson_energy Returns the Josephson energy of the transmon.

        Returns
        -------
        float
            The Josephson energy of the transmon.
        """
        res_freq = self._freq - self._anharm
        joseph_energy = (res_freq / math.sqrt(8 * self.charge_energy)) ** 2

        cos_term = math.cos(math.pi * self.ext_flux)
        shifted_energy = joseph_energy * abs(cos_term)
        return shifted_energy
        
    def _get_raise_op(self) -> Array:
        """
        _get_creation_op Returns the raising (creation) operator of the transmon.

        Returns
        -------
        Array
            The raising (creation) operator of the transmon.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self.dim))
        op = jnp.diag(offdiag, k=-1)
        return op
    
    def get_raise_op(self) -> Array:
        """
        get_creation_op Returns the raising (creation) operator of the transmon.

        Returns
        -------
        Array
            The raising (creation) operator in the current basis of the transmon.
        """
        native_op = self._get_raise_op()
        op = self.process_op(native_op)
        return op
    
    def _get_low_op(self) -> Array:
        """
        _get_low_op Returns the lowering (annihilaton) operator of the transmon.

        Returns
        -------
        Array
            The lowering (annihilaton) operator of the transmon.
        """
        offdiag = jnp.sqrt(jnp.arange(1, self.dim))
        op = jnp.diag(offdiag, k=1)
        return op
    
    def get_low_op(self) -> Array:
        """
        get_low_op Returns the creation operator of the transmon.

        Returns
        -------
        Array
            The lowering (annihilaton) operator in the current basis of the transmon.
        """
        native_op = self._get_low_op()
        op = self.process_op(native_op)
        return op
    
    def _get_num_op(self) -> Array:
        """
        _get_num_op Returns the number operator of the transmon.

        Returns
        -------
        Array
            The number operator of the transmon.
        """
        diagonal = jnp.arange(self._dim)
        op = jnp.diag(diagonal)
        return op
    
    def get_num_op(self) -> Array:
        """
        get_num_op Returns the number operator of the transmon.

        Returns
        -------
        Array
            The number operator in the current basis of the transmon.
        """
        native_op = self._get_low_op()
        op = self.process_op(native_op)
        return op
    
    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Returns the Hamiltonian of the transmon in the charge basis.

        Returns
        -------
        Array
            The Hamiltonian of the transmon expressed in the charge basis.
        """

        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        num_op = self._get_num_op()

        anharm_op = raise_op @ raise_op @ low_op @ low_op

        hamiltonian = self._freq * num_op + 0.5 * self._anharm * anharm_op
        return hamiltonian

    def get_potential(self, phases: Union[float, Array]) -> Array:
        """
        potential Returns the potential energy of the transmon.

        Parameters
        ----------
        phases : Array
            The phase values.

        Returns
        -------
        Array
            The potential energy.
        """
        potential = -self.josephson_energy * jnp.cos(phases)
        return potential
    
    @staticmethod
    def from_energies(
        label: str,
        charge_energy: float,
        joseph_energy: float,
        dim: int = 2,
        ext_flux: float = 0.0,
    ) -> "AnharmonicOscillator":
        """
        from_energies Create an AnharmonicOscillator from the charging and Josephson energies. This is so far assuming symmetric junctions.

        Parameters
        ----------
        label : str
            The label of the transmon.
        charge_energy : float
            The charging energy of the transmon.
        joseph_energy : float
            The Josephson energy of the transmon.
        dim : int, optional
            The dimension of the Hilbert space of the transmon, by default 2
        ext_flux : float, optional
            The external flux through the SQUID loop of the transmon, by default 0

        Returns
        -------
        AnharmonicOscillator
            The AnharmonicOscillator instance.
        """
        anharmonicity = -charge_energy
        res_freq = math.sqrt(8 * charge_energy * joseph_energy)
        frequency = res_freq + anharmonicity
        return AnharmonicOscillator(
            label=label,
            frequency=frequency,
            anharmonicity=anharmonicity,
            ext_flux=ext_flux,
            dim=dim,
        )