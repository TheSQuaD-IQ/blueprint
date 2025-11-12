from jaxtyping import Array
from equinox import field

from ..systems import System
from .drive import Drive, Pulse


class ChargeDrive(Drive):
    """
    Drive module
    """

    label: str = field(static=True, converter=str)
    pulse: Pulse

    def get_drive_op(self, system: System) -> Array:
        return system.get_charge_op()  # type: ignore


class FluxDrive(Drive):
    """
    Drive module
    """

    label: str = field(static=True, converter=str)
    pulse: Pulse

    def get_drive_op(self, system: System) -> Array:
        return system.get_flux_op()  # type: ignore


class CosFluxDrive(Drive):
    """
    Drive module
    """

    label: str = field(static=True, converter=str)
    pulse: Pulse

    def get_drive_op(self, system: System) -> Array:
        return system.get_cosflux_op()  # type: ignore


class SinFluxDrive(Drive):
    """
    Drive module
    """

    label: str = field(static=True, converter=str)
    pulse: Pulse

    def get_drive_op(self, system: System) -> Array:
        return system.get_sinflux_op()  # type: ignore


class DetuningDrive(Drive):
    """
    Drive module
    """

    label: str = field(static=True, converter=str)
    pulse: Pulse

    def get_drive_op(self, system: System) -> Array:
        return system.get_number_op()  # type: ignore
