from __future__ import annotations

from jaxtyping import Array

from .coupling import Coupling, TunableCoupling
from ..systems import System


class ChargeCoupling(Coupling):
    """Capacitive (charge) coupling implementation."""

    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Return capacitive coupling operator between two systems.

        Parameters
        ----------
        system, other_system : System
            Systems participating in the coupling.

        Returns
        -------
        Array
            Coupling operator on joint Hilbert space.
        """
        charge_op = system.get_charge_op()  # type: ignore
        other_charge_op = other_system.get_charge_op()  # type: ignore
        coupling_op = charge_op @ other_charge_op
        return coupling_op


class FluxCoupling(Coupling):
    """Inductive (flux) coupling implementation."""

    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Return inductive coupling operator between two systems.

        Parameters
        ----------
        system, other_system : System
            Systems participating in the coupling.

        Returns
        -------
        Array
            Coupling operator on joint Hilbert space.
        """
        flux_op = system.get_flux_op()  # type: ignore
        other_flux_op = other_system.get_flux_op()  # type: ignore
        coupling_op = flux_op @ other_flux_op
        return coupling_op


class TunableChargeCoupling(TunableCoupling):
    """Tunable capacitive coupling implementation."""

    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Return capacitive coupling operator for tunable coupling.

        Parameters
        ----------
        system, other_system : System
            Systems participating in the coupling.

        Returns
        -------
        Array
            Coupling operator on joint Hilbert space.
        """
        charge_op = system.get_charge_op()  # type: ignore
        other_charge_op = other_system.get_charge_op()  # type: ignore
        coupling_op = charge_op @ other_charge_op
        return coupling_op


class TunableFluxCoupling(TunableCoupling):
    """Tunable inductive coupling implementation."""

    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Return inductive coupling operator for tunable coupling.

        Parameters
        ----------
        system, other_system : System
            Systems participating in the coupling.

        Returns
        -------
        Array
            Coupling operator on joint Hilbert space.
        """
        flux_op = system.get_flux_op()  # type: ignore
        other_flux_op = other_system.get_flux_op()  # type: ignore
        coupling_op = flux_op @ other_flux_op
        return coupling_op
