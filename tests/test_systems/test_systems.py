from abc import abstractmethod

import numpy as np


class BaseTestSystem:
    """Base class for testing systems."""

    @abstractmethod
    def load_system(self):
        """This method should be implemented in the subclass to load the specific system and its test data."""

    def test_hamiltonian_is_hermitian(self, load_system):
        system, _ = load_system
        hamiltonian = system.get_hamiltonian()
        assert np.allclose(hamiltonian, hamiltonian.conj().T)

    def test_hamiltonian(self, load_system):
        system, test_data = load_system
        hamiltonian = system.get_hamiltonian()
        expected_hamiltonian = test_data["hamiltonian"]
        assert np.allclose(hamiltonian, expected_hamiltonian)

    def test_eigenvalues(self, load_system):
        system, test_data = load_system
        eigvals, _ = system.get_eigenstates()
        expected_eigvals = test_data["eigvals"]
        assert np.allclose(eigvals, expected_eigvals)

    def test_charge_matrix_elements(self, load_system):
        system, test_data = load_system
        charge_op = system.get_charge_op()
        expected_charge_op = test_data["charge_op"]
        assert np.allclose(charge_op, expected_charge_op, atol=1e-7)
