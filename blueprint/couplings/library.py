from typing import Union

from .coupling import Coupling
from ..base import QuantumSystem

Numeric = Union[float, complex]


def get_capacitive_coupling(
    label: str,
    qubit: QuantumSystem,
    coupled_qubit: QuantumSystem,
    coupling_strength: Numeric,
) -> Coupling:
    """
    get_capacitive_coupling Returns a capacitive coupling between two qubits.
    This corresponds to a term of the form $H_{coup} = \\eta \\hat{n}_{1}\\hat{n}_{2}$,
    where $\\eta$ corresponds to the coupling strength (`coupling_strength`),
    and $\\hat{n}_{1}$ and $\\hat{n}_{2}$ are the charge operators of the
    two qubits (`qubit` and `coupled_qubit`).

    Parameters
    ----------
    qubit : QuantumSystem
        The first qubit of the pair that are coupled.
    coupled_qubit : QuantumSystem
        The second qubit of the pair that are coupled.
    label : str
        The label of the coupling term.
    coupling_strength : Numeric
        The strength of the capacitive coupling.

    Returns
    -------
    Coupling
        The capacitive coupling between the two qubits.

    Raises
    ------
    AttributeError
        If the qubit does not have a charge operator.
    AttributeError
        If the coupled_qubit does not have a charge operator.
    """
    try:
        qubit_op = qubit.get_charge_op()
    except AttributeError as exc:
        raise AttributeError(f"Qubit {qubit} does not have a charge operator.") from exc

    try:
        coupled_op = coupled_qubit.get_charge_op()
    except AttributeError as exc:
        raise AttributeError(
            f"Qubit {coupled_qubit} does not have a charge operator."
        ) from exc
    operator = qubit_op @ coupled_op
    qubit_labels = (qubit.label, coupled_qubit.label)
    coupling = Coupling(label, operator, coupling_strength, qubit_labels)
    return coupling
