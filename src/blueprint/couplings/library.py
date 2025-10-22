from __future__ import annotations

from jaxtyping import Array

from .coupling import Coupling, TunableCoupling
from ..systems import System


class ChargeCoupling(Coupling):
    """
    ChargeCoupling A class implementing a capacitive coupling term.
    """

    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Returns the coupling operator corresponding to the coupling.

        Parameters
        ----------
        system : System
            The first system that is coupled via the capacitor.
        other_system : System
            The second system that is coupled via the capacitor

        Returns
        -------
        Array
            The coupling operator corresponding to the coupling.
        """
        charge_op = system.get_charge_op()  # type: ignore
        other_charge_op = other_system.get_charge_op()  # type: ignore
        coupling_op = charge_op @ other_charge_op
        return coupling_op


class FluxCoupling(Coupling):
    """
    FluxCoupling A class implementing an inductive coupling term.
    """

    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Returns the coupling operator corresponding to the coupling.

        Parameters
        ----------
        system : System
            The first system that is coupled via the inductor.
        other_system : System
            The second system that is coupled via the inductor.

        Returns
        -------
        Array
            The coupling operator corresponding to the coupling.
        """
        flux_op = system.get_flux_op()  # type: ignore
        other_flux_op = other_system.get_flux_op()  # type: ignore
        coupling_op = flux_op @ other_flux_op
        return coupling_op


class TunableChargeCoupling(TunableCoupling):
    """
    TunableChargeCoupling A class implementing a tunable capacitive coupling term.
    """

    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Returns the coupling operator corresponding to the coupling.

        Parameters
        ----------
        system : System
            The first system that is coupled via the capacitor.
        other_system : System
            The second system that is coupled via the capacitor

        Returns
        -------
        Array
            The coupling operator corresponding to the coupling.
        """
        charge_op = system.get_charge_op()  # type: ignore
        other_charge_op = other_system.get_charge_op()  # type: ignore
        coupling_op = charge_op @ other_charge_op
        return coupling_op


class TunableFluxCoupling(TunableCoupling):
    """
    TunableFluxCoupling A class implementing a tunable inductive coupling term.
    """

    def get_coupling_op(self, system: System, other_system: System) -> Array:
        """
        get_coupling_op Returns the coupling operator corresponding to the coupling.

        Parameters
        ----------
        system : System
            The first system that is coupled via the inductor.
        other_system : System
            The second system that is coupled via the inductor.

        Returns
        -------
        Array
            The coupling operator corresponding to the coupling.
        """
        flux_op = system.get_flux_op()  # type: ignore
        other_flux_op = other_system.get_flux_op()  # type: ignore
        coupling_op = flux_op @ other_flux_op
        return coupling_op
