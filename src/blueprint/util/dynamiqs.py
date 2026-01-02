from dynamiqs import TimeQArray

from ..systems import System
from ..drives import BaseDrive as Drive


def get_system_hamiltonian(system: System, *drives: Drive) -> TimeQArray:
    """
    get_system_hamiltonian Return the system Hamiltonian as a dynamiqs TimeQArray.

    Parameters
    ----------
    system : System
        Quantum system.

    Returns
    -------
    modulated.TimeQArray
        Time-dependent Hamiltonian for the system.
    """
    hamiltonian = system.get_hamiltonian_qarray()

    for drive in drives:
        drive_hamiltonian = drive.get_hamiltonian_qarray(system)
        hamiltonian = hamiltonian + drive_hamiltonian

    return hamiltonian
