from jax import scipy as jsc 
from jax import numpy as jnp
from jax import Array 

from ..base import QuantumSystem

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
        
class Fluxonium(QuantumSystem):
    """
    Fluxonium qubit model.

    Args: 
        josephson_energy: The josephson energy of the fluxonium black sheep junction 
        charging_energy: The charging energy of the fluxonium 
        ext_flux: The external flux through the fluxonium loop
        fock_cut_off: The number of fock states to consider
    """

    def __init__(
        self,
        label: str, 
        josephson_energy: float,
        charging_energy: float,
        inductive_energy: float,
        ext_flux: float,
        fock_cut_off: int,
    ) -> None: 
        
        # The fluxonium parameters (Josephson energy, charging energy, external flux)
        check_var_validity(josephson_energy, "josephson_energy", min_value=0.0)
        self._ej = josephson_energy

        check_var_validity(charging_energy, "charging_energy", min_value=0.0)
        self._ec = charging_energy

        check_var_validity(inductive_energy, "inductive_energy", min_value=0.0)
        self._el = inductive_energy

        check_var_validity(ext_flux, "ext_flux")
        self._ext_flux = ext_flux

        # The number of charge states to consider when constructing the Hamiltonian/operators
        # in the native (charge) basis.
        if not isinstance(fock_cut_off, int):
            raise ValueError(
                f"The charge cutoff expected to be an integer, "
                f"instead got type {type(fock_cut_off)}."
            )
        if fock_cut_off <= 0:
            raise ValueError(
                "The charge cutoff must be a non-negative integer or equal to zero."
            )
        self._fockcut: int = fock_cut_off

        # The dimension of the Hilbert space
        dim: int = self._fockcut
        super().__init__(label, dim)

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
    def charge_cutoff(self) -> int:
        """
        charge_cutoff Returns the number of charge states to consider.

        Returns
        -------
        int
            The number of charge states to consider.
        """
        return self._fockcut
    
    @property
    def charge_zpf(self) -> float:
        """
        charge_zpf Returns the zero-point fluctuations of the charge.

        Returns
        -------
        float
            The zero-point fluctuations of the charge.
        """
        return (1.j / jnp.sqrt(2)) * (self.inductive_energy / (8 * self.charging_energy)) ** 0.25

    @property
    def flux_zpf(self) -> float:
        """
        flux_zpf Returns the zero-point fluctuations of the flux.

        Returns
        -------
        float
            The zero-point fluctuations of the flux.
        """
        return (1 / jnp.sqrt(2)) * (8 * self.charging_energy / self.inductive_energy) ** 0.25
    
    def _get_raise_op(self) -> Array:
        """
        _get_raise_op Returns the raising (creation) operator of the fluxonium.

        Returns
        -------
        Array
            The raising operator, in the Fock basis.
        """
        dim = self._trunc_dim or self.dim 
        offdiag = jnp.sqrt(jnp.arange(1, dim))
        return jnp.diag(offdiag, k=-1)

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
        dim = self._trunc_dim or self._dim
        offdiag = jnp.sqrt(jnp.arange(1, dim))
        return jnp.diag(offdiag, k=1)

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

    def _get_charge_op(self) -> Array:
        """
        _get_charge_op Returns the charge operator of the fluxonium.

        Returns
        -------
        Array
            The charge operator, in the Fock basis.
        """
        op = self.charge_zpf * (self._get_raise_op() - self._get_low_op())
        return op
    
    def get_charge_op(self) -> Array:
        """
        get_charge_op Returns the charge operator of the fluxonium.

        Returns
        -------
        Array
            The charge operator, in the current basis of the transmon.
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
        op = self.flux_zpf * (self._get_raise_op() + self._get_low_op())
        return op
    
    def get_flux_op(self) -> Array:
        """
        get_flux_op Returns the flux operator of the fluxonium.

        Returns
        -------
        Array
            The flux operator, in the current basis of the transmon.
        """
        flux_op = self._get_flux_op()
        processed_op = self.process_op(flux_op)
        return processed_op
    
    def _get_cosphi_op(self) -> Array: 
        """
        _get_cosphi_op Returns the cos(phi) operator of the fluxonium in the fock basis.

        Returns
        -------
        Array
            The cos(phi) operator of the fluxonium, in the Fock basis.
        """
        exponent = 1.j * self._get_flux_op()
        op = 0.5 * (jsc.linalg.expm(exponent) + jsc.linalg.expm(-exponent))
        return op
    
    def get_cosphi_op(self) -> Array:
        """
        get_cosphi_op Returns the cos(phi) operator of the fluxonium.

        Returns
        -------
        Array
            The cos(phi) operator of the fluxonium, in the current basis of the fluxonium.
        """
        cosphi_op = self._get_cosphi_op()
        processed_op = self.process_op(cosphi_op)
        return processed_op
    
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
        cosphi_op = self._get_cosphi_op()
        phi_op = self._get_flux_op()

        dim = self._trunc_dim or self.dim
        id_op = jnp.identity(dim)

        inductive_op = phi_op + (id_op * (2 * jnp.pi * self._ext_flux))

        potential_term = (0.5 * self._el * (inductive_op @ inductive_op)) - self._ej * cosphi_op
        return potential_term
    
    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Returns the Hamiltonian of the fluxonium in the fock basis.

        Returns
        -------
        Array
            The Hamiltonian of the fluxonium, in the Fock basis.
        """
        hamil = self._get_kinetic_term() + self._get_potential_term()
        return hamil
    
    def get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the fluxonium.

        Returns
        -------
        Array
            The number operator of the fluxonium.

        Raises
        ------
        NotImplementedError
            If the fluxonium is not diagonalized.
        """
        if self._diagonalized:
            diag_elems = jnp.arange(self._dim)
            num_op = jnp.diag(diag_elems)
            processed_op = self.process_op(num_op, diagonalize=False)
            return processed_op

        raise NotImplementedError(
            "The number operator is only available in the diagonal (energy) basis."
        )
    