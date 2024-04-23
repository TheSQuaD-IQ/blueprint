import math
from typing import Union, Tuple, Callable
from functools import wraps

from scipy.optimize import minimize
from jax import numpy as jnp
from jax import Array

from ..base import QuantumSystem
from ..drives import Drive


def check_var_validity(
    arg: float,
    argname: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """
    check_valid_variable Checks if the provided variable is a valid numeric variable.

    """
    if not isinstance(arg, float):
        raise ValueError(
            f"The {argname} is expected to be a float, instead got type {type(arg)}."
        )

    if min_value is not None:
        if arg < min_value:
            raise ValueError(
                f"The {argname} must be greater than or equal to {min_value}."
            )
    if max_value is not None:
        if arg > max_value:
            raise ValueError(
                f"The {argname} must be less than or equal to {max_value}."
            )


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
        check_var_validity(charging_energy, "charging_energy", min_value=0.0)
        self._ec = charging_energy

        check_var_validity(josephson_energy, "josephson_energy", min_value=0.0)
        self._ej = josephson_energy

        check_var_validity(offset_charge, "offset_charge")
        self._ng = offset_charge

        check_var_validity(ext_flux, "ext_flux")
        self._ext_flux = ext_flux

        check_var_validity(asymmetry, "asymmetry", min_value=0.0, max_value=1.0)
        self._asymm = asymmetry

        # The number of charge states to consider when constructing the Hamiltonian/operators
        # in the native (charge) basis.
        if not isinstance(charge_cutoff, int):
            raise ValueError(
                f"The charge cutoff expected to be an integer, "
                f"instead got type {type(charge_cutoff)}."
            )
        if charge_cutoff <= 0:
            raise ValueError(
                "The charge cutoff must be a non-negative integer or equal to zero."
            )
        self._ncut: int = charge_cutoff

        # The dimension of the Hilbert space
        dim: int = 2 * self._ncut + 1
        super().__init__(label, dim)

        # The relaxation and dephasing times
        if relax_time is not None:
            check_var_validity(relax_time, "relax_time", min_value=0.0)
        self._relax_time = relax_time

        if deph_time is not None:
            check_var_validity(deph_time, "deph_time", min_value=0.0)
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

    def get_josephson_energy(self, ext_flux: float) -> float:
        """
        eff_josephson_energy Returns the effective Josephson energy of the transmon.
        This is the Josephson energy modified by the external flux and the junction asymmetry.

        Returns
        -------
        float
            The effective Josephson energy
        """
        cos_term = math.cos(ext_flux)
        sqrt_term = math.sqrt(1 + self._asymm**2 * math.tan(ext_flux) ** 2)

        prefactor = abs(cos_term) * sqrt_term
        return self._ej * prefactor

    @property
    def eff_josephson_energy(self) -> float:
        """
        eff_josephson_energy Returns the effective Josephson energy of the transmon.
        This is the Josephson energy modified by the external flux and the junction asymmetry.

        Returns
        -------
        float
            The effective Josephson energy.
        """
        return self.get_josephson_energy(self._ext_flux)

    def ext_flux_to_approx_freq(self, ext_flux: float) -> float:
        """
        ext_flux_to_approx_freq Returns the approximate 0-1 frequency of the transmon.

        Returns
        -------
        float
            The approximate transmon 0-1 frequency.
        """
        sqrt_term = math.sqrt(8 * self._ec * self.get_josephson_energy(ext_flux))
        return sqrt_term - self._ec

    @property
    def _approx_freq(self) -> float:
        """
        _approx_freq Returns the approximate 0-1 frequency of the transmon.

        Returns
        -------
        float
            The approximate transmon 0-1 frequency.
        """
        return self.ext_flux_to_approx_freq(self._ext_flux)

    @property
    def approximate_frequency(self) -> float:
        """
        approximate_frequency Returns the approximate 0-1 frequency of the transmon.

        Returns
        -------
        float
            The approximate transmon 0-1 frequency.
        """
        return self._approx_freq

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
        if not self.is_diagonalized:
            raise ValueError(
                "The charge zero-point fluctuations are only available in the diagonal basis."
            )
        charge_op = self._get_charge_op()
        diag_op = self.process_op(
            charge_op, diagonalize=True, embed=False, truncate=False
        )
        squared_op = diag_op @ diag_op
        exp_val = squared_op[0, 0]
        charge_fluctuations = math.sqrt(float(exp_val.real))
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
        return 1 / (2 * self.charge_zpf)

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
        charge_op = self._get_charge_op()
        processed_op = self.process_op(charge_op)
        return processed_op

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
        sinphi_op = self._get_sinphi_op()

        cos_term = jnp.cos(self._ext_flux) * cosphi_op
        sin_term = self._asymm * jnp.sin(self._ext_flux) * sinphi_op

        potential_term = -self._ej * (cos_term + sin_term)
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
        potential = -self.eff_josephson_energy * jnp.cos(phases)
        return potential

    def add_flux_drive(self, label: str, flux_pulse: Callable) -> None:
        """
        add_flux_drive Applies a flux drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        flux_pulse : Callable
            The time-dependent flux pulse applied to the transmon. This must be a
            callable object that returns the applied flux pulse as a function of time.

        Raises
        ------
        ValueError
            If a drive with the same label has already been
        """
        if label in self._drives:
            raise ValueError(
                f"A drive with the label '{label}' has already been applied to the transmon."
            )

        if not isinstance(flux_pulse, Callable):
            raise ValueError(
                f"The flux pulse must be either a float or a Callable object, instead got type {type(flux_pulse)}."
            )

        @wraps(flux_pulse)
        def cos_prefactor(*args, **kwargs) -> Array:
            applied_flux = flux_pulse(*args, **kwargs)
            cur_flux = self._ext_flux + applied_flux
            prefactor = -self._ej * (jnp.cos(cur_flux) - jnp.cos(self._ext_flux))
            return prefactor

        @wraps(flux_pulse)
        def sin_prefactor(*args, **kwargs) -> Array:
            applied_flux = flux_pulse(*args, **kwargs)
            cur_flux = self._ext_flux + applied_flux
            prefactor = (
                -self._ej * self._asymm * (jnp.sin(cur_flux) - jnp.sin(self._ext_flux))
            )
            return prefactor

        prefactors = (cos_prefactor, sin_prefactor)

        cosphi_op = self._get_cosphi_op()
        sinphi_op = self._get_sinphi_op()
        ops = (cosphi_op, sinphi_op)

        drive = Drive(label, prefactors, ops)
        self._drives[label] = drive

    def add_charge_drive(self, label: str, charge_pulse: Callable) -> None:
        """
        add_charge_drive Applies a charge drive to the transmon.

        Parameters
        ----------
        label : str
            The label of the drive.
        charge_pulse : Callable
            The time-dependent charge pulse applied to the transmon. This must be a
            callable object that returns the applied charge pulse as a function of time.

        Raises
        ------
        ValueError
            If a drive with the same label has already been
        """
        if label in self._drives:
            raise ValueError(
                f"A drive with the label '{label}' has already been applied to the transmon."
            )

        if not isinstance(charge_pulse, Callable):
            raise ValueError(
                f"The charge pulse must be either a float or a Callable object, instead got type {type(charge_pulse)}."
            )

        charge_op = self._get_charge_op()
        drive = Drive(label, charge_pulse, charge_op)
        self._drives[label] = drive

    @staticmethod
    def from_params(
        label: str,
        frequency: float,
        anharmonicity: float,
        ext_flux: float = 0.0,
        offset_charge: float = 0.0,
        asymmetry: float = 0.0,
        charge_cutoff: int = 100,
        relax_time: float | None = None,
        deph_time: float | None = None,
    ) -> "TunableTransmon":
        """
        from_params Create a TunableTransmon based on the qubit frequency and anharmonicity. This function also accepts other parameters, such as the external flux, offset charge, and junction asymmetry. The function will optimize the charging and Josephson energies to match the provided frequency and anharmonicity. The optimization is done using the scipy.optimize.minimize function. The optimization is done in the following way:
        1. Calculate the initial guesses for the maximum Josephson energy and Charging energy based on the provided frequency and anharmonicity.
        2. Define an objective function that calculates the difference between the provided frequency and anharmonicity and the calculated frequency and anharmonicity based on the charging and Josephson energies. The objective function is the sum of the squared differences between the provided and calculated values. The function constructs the the transmon Hamiltonian in the charge basis and calculates the frequency and anharmonicity based on the eigenvalues.
        3. Optimize the objective function to find the charging and Josephson energies that best match the provided frequency and anharmonicity.

        Parameters
        ----------
        label : str
            The label of the transmon.
        frequency : float
            The frequency of the transmon.
        anharmonicity : float
            The anharmonicity of the transmon.
        ext_flux : float, optional
            The external flux applied to the transmon, by default 0.0
        offset_charge : float, optional
            The offset charge, by default 0.0
        asymmetry : float, optional
            The SQUID junction asymmetry, by default 0.0
        charge_cutoff : int, optional
            _description_, by default 100
        relax_time : float | None, optional
            The relaxation time of the transmon, by default None
        deph_time : float | None, optional
            The dephasing time of the transmon, by default None

        Returns
        -------
        TunableTransmon
            The TunableTransmon instance.

        Raises
        ------
        ValueError
            If the frequency is not a float.
        ValueError
            If the anharmonicity is not a float.
        ValueError
            If the anharmonicity is positive.
        ValueError
            If the optimization fails.
        """
        if not isinstance(frequency, float):
            raise ValueError(
                f"The maximum frequency expected to be a float, instead got type {type(frequency)}."
            )
        if not isinstance(anharmonicity, float):
            raise ValueError(
                f"The anharmonicity expected to be a float, instead got type {type(anharmonicity)}."
            )
        if anharmonicity > 0:
            raise ValueError(
                "The anharmonicity is expected to be negative for a transmon qubits. Instead a positive anharmonicity was provided."
            )

        init_ec = -anharmonicity
        init_ej = (frequency + init_ec) / (8 * init_ec)

        cos_term = math.cos(ext_flux)
        sqrt_term = math.sqrt(1 + asymmetry**2 * math.tan(ext_flux) ** 2)
        prefactor = abs(cos_term) * sqrt_term
        max_ej = init_ej / prefactor

        def objective_func(x: Tuple[float, float]) -> Array:
            charging_energy, josephson_energy = x
            transmon = TunableTransmon(
                label,
                charging_energy,
                josephson_energy,
                offset_charge,
                ext_flux,
                asymmetry,
                charge_cutoff,
            )
            freq_diff = frequency - transmon.frequency
            anharm_diff = anharmonicity - transmon.anharmonicity
            result = freq_diff**2 + anharm_diff**2
            return result

        init_guess = (max_ej, init_ec)

        result = minimize(objective_func, init_guess)
        if not result.success:
            raise ValueError(f"Optimization failed with message: {result.message}.")

        charging_energy, josephson_energy = result.x
        return TunableTransmon(
            label,
            charging_energy,
            josephson_energy,
            offset_charge,
            ext_flux,
            asymmetry,
            charge_cutoff,
            relax_time,
            deph_time,
        )
