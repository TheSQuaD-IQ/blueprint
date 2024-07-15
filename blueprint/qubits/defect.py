from __future__ import annotations
import math
from typing import Iterator

from jax import numpy as jnp
from jax import Array

from ..base import QuantumSystem


class TLSDefect(QuantumSystem):
    """
    TLSDefect A model for a two-level system (TLS) defect.
    """

    def __init__(
        self,
        label: str,
        frequency: float,
        decay_rate: float | None = None,
        deph_rate: float | None = None,
        thermal_photons: float = 0.0,
        *,
        dim: int = 2,
    ) -> None:
        if not isinstance(frequency, float):
            raise ValueError(
                f"The frequency must be a float, instead got type {type(frequency)}."
            )
        if frequency <= 0.0:
            raise ValueError("The frequency must be greater than zero.")
        self._freq = frequency

        super().__init__(label, dim)

        # The relaxation and pure dephasing rates
        if decay_rate is not None:
            if not isinstance(decay_rate, float):
                raise ValueError(
                    f"The decay rate must be a float, instead got type {type(decay_rate)}."
                )
            if decay_rate < 0.0:
                raise ValueError("The decay rate must be greater than zero.")
        self._decay_rate = decay_rate

        if deph_rate is not None:
            if not isinstance(deph_rate, float):
                raise ValueError(
                    f"The dephasing rate must be a float, instead got type {type(deph_rate)}."
                )
            if deph_rate < 0.0:
                raise ValueError("The dephasing rate must be greater than zero.")
        self._deph_rate = deph_rate

        if not isinstance(thermal_photons, float):
            raise ValueError(
                f"The thermal photons must be a float, instead got type {type(thermal_photons)}."
            )
        if thermal_photons < 0.0:
            raise ValueError("The thermal photons must be greater than zero.")
        self._n_thermal = thermal_photons

    @property
    def frequency(self) -> float:
        """
        frequency Returns the frequency of the TLS defect.

        Returns
        -------
        float
            The frequency of the TLS defect.
        """
        return self._freq

    @frequency.setter
    def frequency(self, frequency: float) -> None:
        """
        frequency Sets the frequency of the TLS defect.

        Parameters
        ----------
        frequency : float
            The frequency of the TLS defect.

        Raises
        ------
        ValueError
            If the frequency is not a float.
        ValueError
            If the frequency is less than or equal to zero.
        """
        if not isinstance(frequency, float):
            raise ValueError(
                f"The frequency must be a float, instead got type {type(frequency)}."
            )
        if frequency <= 0.0:
            raise ValueError("The frequency must be greater than zero.")
        self._freq = frequency

    def _get_raise_op(self) -> Array:
        """
        _get_creation_op Returns the raising (creation) operator of the TLS defect expressed in the native Fock basis.

        Returns
        -------
        Array
            The raising (creation) operator of the TLS defect.
        """
        off_diagonal = jnp.sqrt(jnp.arange(1, self._dim))
        raise_op = jnp.diag(off_diagonal, k=-1)
        return raise_op

    def get_raise_op(self) -> Array:
        """
        get_creation_op Returns the raising (creation) operator of the TLS defect.
        Natively, the raising operator is expressed in the Fock basis.
        If the qubit is diagonalized, truncated, or embedded in a device this operator will be transformed accordingly.

        Returns
        -------
        Array
            The raising (creation) operator in the current basis of the TLS defect.
        """
        raise_op = self._get_raise_op()
        return self.process_op(raise_op)

    def _get_low_op(self) -> Array:
        """
        _get_low_op Returns the lowering (annihilaton) operator of the TLS defect expressed in the native Fock basis.

        Returns
        -------
        Array
            The lowering (annihilaton) operator of the TLS defect.
        """
        off_diagonal = jnp.sqrt(jnp.arange(1, self._dim))
        low_op = jnp.diag(off_diagonal, k=1)
        return low_op

    def get_low_op(self) -> Array:
        """
        get_low_op Returns the lowering (annihilaton) operator of the TLS defect.
        Natively, the lowering operator is expressed in the Fock basis.
        If the qubit is diagonalized, truncated, or embedded in a device this operator will be transformed accordingly.

        Returns
        -------
        Array
            The lowering (annihilaton) operator in the current basis of the TLS defect.
        """
        low_op = self._get_low_op()
        return self.process_op(low_op)

    def _get_number_op(self) -> Array:
        """
        _get_number_op Returns the number operator of the TLS defect.

        Returns
        -------
        Array
            The number operator of the TLS defect.
        """
        diagonal = jnp.arange(self._dim)
        number_op = jnp.diag(diagonal)
        return number_op

    def get_number_op(self) -> Array:
        """
        get_number_op Returns the number operator of the TLS defect expressed in the native Fock basis.
        Natively, the number operator is expressed in the Fock basis.
        If the qubit is diagonalized, truncated, or embedded in a device this operator will be transformed accordingly.

        Returns
        -------
        Array
            The number operator in the current basis of the TLS defect.
        """
        number_op = self._get_number_op()
        return self.process_op(number_op)

    def _get_hamiltonian(self) -> Array:
        """
        _get_hamiltonian Returns the Hamiltonian of the TLS defect in the native Fock basis.

        Returns
        -------
        Array
            The Hamiltonian of the TLS defect in the Fock basis.
        """
        number_op = self._get_number_op()
        hamiltonian = self.frequency * number_op
        return hamiltonian

    def _get_charge_op(self) -> Array:
        """
        _get_charge_op Returns the charge operator of the TLS defect.

        Returns
        -------
        Array
            The charge operator of the TLS defect.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        charge_op = 1.0j * (raise_op - low_op)
        return charge_op

    def get_charge_op(self) -> Array:
        """
        charge_op Returns the (transformed) charge operator of the TLS defect.
        Natively, the charge operator is expressed in the Fock basis.
        If the qubit is diagonalized, truncated, or embedded in a device this operator will be transformed accordingly.

        Returns
        -------
        Array
            The (transformed) charge operator, in the current basis of the TLS defect.
        """
        charge_op = self._get_charge_op()
        return self.process_op(charge_op)

    def _get_flux_op(self) -> Array:
        """
        _get_flux_op Returns the flux operator of the TLS defect in the Fock basis.

        Returns
        -------
        Array
            The flux operator of the TLS defect, expressed in the Fock basis.
        """
        low_op = self._get_low_op()
        raise_op = self._get_raise_op()
        flux_op = raise_op + low_op
        return flux_op

    def get_flux_op(self) -> Array:
        """
        get_flux_op Returns the flux operator of the TLS defect.
        Natively, the flux operator is expressed in the Fock basis.
        If the qubit is diagonalized, truncated, or embedded in a device this operator will be transformed accordingly.

        Returns
        -------
        Array
            The flux operator, in the current basis of the TLS defect.
        """
        flux_op = self._get_flux_op()
        return self.process_op(flux_op)

    def _get_decay_ops(self) -> Iterator[Array]:
        """
        _get_decay_ops Yields the decay (and excitation) jump operators of the TLS defect in the native fock bais.

        Yields
        ------
        Iterator[Array]
            The decay (and excitation) jump operators of the TLS defect.

        Raises
        ------
        ValueError
            If the decay rate of the TLS defect has not been set.
        """
        if self._decay_rate is None:
            raise ValueError("The decay rate of the TLS defect has not been set.")

        decay_prefactor = math.sqrt(self._decay_rate * (1 + self._n_thermal))
        low_op = self._get_low_op()

        decay_op = decay_prefactor * low_op
        yield decay_op

        if self._n_thermal > 0.0:
            exc_prefactor = math.sqrt(self._n_thermal * self._decay_rate)
            raise_op = self._get_raise_op()

            exc_op = exc_prefactor * raise_op
            yield exc_op

    def get_decay_ops(self) -> Iterator[Array]:
        """
        get_decay_ops Yield the decay (and excitation) jump operators of the TLS defect, expressed in the transformed/truncated basis of the TLS defect.

        Yields
        ------
        Iterator[Array]
            The decay (and excitation) jump operators of the TLS defect.
        """
        decay_ops = self._get_decay_ops()
        for decay_op in decay_ops:
            yield self.process_op(decay_op)

    def _get_deph_ops(self) -> Iterator[Array]:
        """
        _get_deph_ops Yields the dephasing jump operators of the TLS defect in the native fock basis.

        Yields
        ------
        Iterator[Array]
            The dephasing jump operators of the TLS defect.

        Raises
        ------
        ValueError
            If the dephasing rate of the TLS defect has not been set.
        """
        if self._deph_rate is None:
            raise ValueError("The deph rate of the TLS defect has not been set.")

        prefactor = math.sqrt(2 * self._deph_rate)
        number_op = self._get_number_op()

        deph_op = prefactor * number_op
        yield deph_op

    def get_deph_ops(self) -> Iterator[Array]:
        """
        get_deph_ops Yield the dephasing jump operators of the TLS defect, expressed in the transformed/truncated basis of the TLS defect.

        Yields
        ------
        Iterator[Array]
            The dephasing jump operators of the TLS defect.
        """
        deph_ops = self._get_deph_ops()
        for deph_op in deph_ops:
            yield self.process_op(deph_op)

    def get_jump_ops(self) -> Iterator[Array]:
        """
        get_jump_ops Yields the jump operators associated with the TLS defect.
        These correspond to either or both the energy relaxation and dephasing processes, depending on whether the values of the relaxation and dephasing times were provided, respectively.

        Yields
        ------
        Iterator[Array]
            The jump operators associated with the TLS defect.
        """
        if self._decay_rate is not None:
            decay_ops = self.get_decay_ops()
            yield from decay_ops

        if self._deph_rate is not None:
            deph_ops = self.get_deph_ops()
            yield from deph_ops
