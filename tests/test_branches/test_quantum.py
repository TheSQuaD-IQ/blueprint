from pathlib import Path

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from blueprint.branches.quantum import get_branches
from blueprint.couplings import ChargeCoupling
from blueprint.systems import Resonator, Transmon
from blueprint.util.quantum import expectation_value

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def load_test_data():
    DATA_DIR = Path.cwd() / "tests/test_branches/data"
    filename = "quantum_branch_analysis_testdata.npz"
    data = np.load(DATA_DIR / filename)
    return data


@pytest.fixture
def build_system(load_test_data):
    # Function for building the necessary objects to pass into get_branches
    data = load_test_data

    charging_energy = data["CHARGING_ENERGY"]
    josephson_energy = data["JOSEPHSON_ENERGY"]
    offset_charge = data["OFFSET_CHARGE"]
    charge_cutoff = int(data["CHARGE_CUTOFF"])
    transmon_dim = int(data["TRANSMON_DIM"])

    res_frequency = data["RES_FREQUENCY"]
    res_impedance = data["RES_IMPEDANCE"]
    res_dim = int(data["RES_DIM"])

    coupling_label = str(data["COUP_LABEL"])
    coupling_strength = data["COUP_STRENGTH"]

    transmon = Transmon(
        label="transmon",
        charging_energy=charging_energy,
        josephson_energy=josephson_energy,
        offset_charge=offset_charge,
        charge_cutoff=charge_cutoff,
        dim=transmon_dim,
    )

    resonator = Resonator.from_frequency(
        label="resonator",
        frequency=res_frequency,
        impedance=res_impedance,
        dim=int(res_dim),
    )

    device_dims = (transmon_dim, res_dim)

    coupling = ChargeCoupling(
        label=coupling_label,
        strength=coupling_strength,
    )

    _, transmon_states = transmon.get_eigenstates()
    _, res_states = resonator.get_eigenstates()
    res_ground_state = res_states[:, 0]

    prod_states = jnp.kron(transmon_states, res_ground_state[:, None])

    transmon = transmon.embed(0, device_dims)
    resonator = resonator.embed(1, device_dims)

    transmon_hamiltonian = transmon.get_hamiltonian()
    resonator_hamiltonian = resonator.get_hamiltonian()
    coupling_hamiltonian = coupling.get_hamiltonian(transmon, resonator)

    hamiltonian = transmon_hamiltonian + resonator_hamiltonian + coupling_hamiltonian

    raise_op = resonator.get_raise_op()

    return hamiltonian, raise_op, prod_states, transmon, resonator


def test_get_branches(build_system, load_test_data):

    comparison_cutoff = 20  # NOTE: To not compare the last X values in the resonator axis as it can be unstable

    hamiltonian, raise_op, prod_states, transmon, resonator = build_system
    energies, branches = get_branches(
        hamiltonian=hamiltonian, raise_op=raise_op, prod_states=prod_states
    )

    transmon_number_op = transmon.get_number_op()
    resonator_number_op = resonator.get_number_op()

    res_populations = jnp.real(expectation_value(branches, resonator_number_op))
    transmon_populations = jnp.real(expectation_value(branches, transmon_number_op))

    energies = energies.reshape(transmon.dim, resonator.dim)
    res_populations = res_populations.reshape(transmon.dim, resonator.dim)
    transmon_populations = transmon_populations.reshape(transmon.dim, resonator.dim)

    actual_energies = energies[:, :-comparison_cutoff]
    actual_transmon_populations = transmon_populations[:, :-comparison_cutoff]
    actual_res_populations = res_populations[:, :-comparison_cutoff]

    expected_energies = load_test_data["energies"][:, :-comparison_cutoff]
    expected_transmon_populations = load_test_data["transmon_populations"][
        :, :-comparison_cutoff
    ]
    expected_res_populations = load_test_data["res_populations"][:, :-comparison_cutoff]

    # NOTE: Successive assertions for now, can change later
    assert np.allclose(actual_energies, expected_energies)
    assert np.allclose(actual_transmon_populations, expected_transmon_populations)
    assert np.allclose(actual_res_populations, expected_res_populations)
