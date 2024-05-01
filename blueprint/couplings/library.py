from typing import Union

from .coupling import Coupling
from ..base import QuantumSystem

Numeric = Union[float, complex]


def get_capacitive_coupling(
    label: str,
    system: QuantumSystem,
    coupled_system: QuantumSystem,
    coupling_strength: Numeric,
) -> Coupling:
    """
    get_capacitive_coupling Returns a capacitive coupling between two systems (qubits or resonators).
    This corresponds to a term of the form $H_{coup} = \\eta \\hat{n}_{1}\\hat{n}_{2}$,
    where $\\eta$ corresponds to the coupling strength (`coupling_strength`),
    and $\\hat{n}_{1}$ and $\\hat{n}_{2}$ are the charge operators of the
    two systems (`system` and `coupled_system`).

    Parameters
    ----------
    system : QuantumSystem
        The first quantum system of the pair that are coupled.
    coupled_system : QuantumSystem
        The second quantum system of the pair that are coupled.
    label : str
        The label of the coupling term.
    coupling_strength : Numeric
        The strength of the capacitive coupling.

    Returns
    -------
    Coupling
        The capacitive coupling between the two systems.

    Raises
    ------
    AttributeError
        If the system does not have a charge operator.
    AttributeError
        If the coupled_system does not have a charge operator.
    """
    try:
        system_op = system.get_charge_op()
    except AttributeError as exc:
        raise AttributeError(
            f"system {system} does not have a charge operator."
        ) from exc

    try:
        coupled_op = coupled_system.get_charge_op()
    except AttributeError as exc:
        raise AttributeError(
            f"system {coupled_system} does not have a charge operator."
        ) from exc
    operator = system_op @ coupled_op
    system_labels = (system.label, coupled_system.label)
    coupling = Coupling(label, operator, coupling_strength, system_labels)
    return coupling
