import math
from pathlib import Path

import jax
import numpy as np
import pytest
from dynamiqs.method import Tsit5
from jax import numpy as jnp
from scipy.constants import e, hbar

from blueprint.branches.floquet import get_branches
from blueprint.drives.pulses import get_cos_pulse
from blueprint.systems import Transmon

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def load_test_data():
    DATA_DIR = Path.cwd() / "tests/test_branches/data"
    filename = "floquet_branch_analysis_testdata.npz"
    data = np.load(DATA_DIR / filename)
    return data


def get_res_charge_zpf(frequency: float, impedance: float) -> float:
    capacitance = 1 / (impedance * frequency)

    redifined_e = e / math.sqrt(hbar)
    charging_energy = (redifined_e**2) / (2 * capacitance)

    inductance = impedance / frequency
    inductive_energy = 1 / (4 * (redifined_e**2) * inductance)

    charge_zpf = (inductive_energy / (32 * charging_energy)) ** 0.25

    return charge_zpf


@pytest.fixture
def build_system(load_test_data):
    # Function for building the necessary objects to pass into get_branches
    data = load_test_data

    charging_energy = data["CHARGING_ENERGY"]
    josephson_energy = data["JOSEPHSON_ENERGY"]
    offset_charge = data["OFFSET_CHARGE"]
    charge_cutoff = int(data["CHARGE_CUTOFF"])
    transmon_dim = int(data["TRANSMON_DIM"])

    res_frequency = data["RES_FREQ"]
    res_impedance = data["RES_IMPEDANCE"]

    coupling_strength = data["COUP_STRENGTH"]

    MIN_NUM_PHOTONS = int(data["MIN_NUM_PHOTONS"])
    MAX_NUM_PHOTONS = int(data["MAX_NUM_PHOTONS"])
    NUM_PHOTONS = int(data["NUM_PHOTONS"])

    DRIVE_PHASE = float(data["DRIVE_PHASE"])
    DRIVE_FREQ = float(data["DRIVE_FREQ"])
    DRIVE_PERIOD = 2 * math.pi / DRIVE_FREQ

    transmon = Transmon(
        label="transmon",
        charging_energy=charging_energy,
        josephson_energy=josephson_energy,
        offset_charge=offset_charge,
        charge_cutoff=charge_cutoff,
        dim=transmon_dim,
    )

    hamiltonian = transmon.get_hamiltonian()
    hamiltonian = hamiltonian.astype(complex)

    photons = jnp.linspace(MIN_NUM_PHOTONS, MAX_NUM_PHOTONS + 1, NUM_PHOTONS)

    res_charge_zpf = get_res_charge_zpf(res_frequency, res_impedance)
    drive_amplitudes = 2 * coupling_strength * res_charge_zpf * jnp.sqrt(photons)

    drive_op = transmon.get_charge_op()
    drive_pulse = get_cos_pulse(drive_amplitudes, DRIVE_FREQ, DRIVE_PHASE)

    method = Tsit5(rtol=1e-8, atol=1e-8, max_steps=1000000)

    return hamiltonian, drive_pulse, drive_op, DRIVE_PERIOD, method, transmon


def test_get_branches(build_system, load_test_data):
    comparison_cutoff = 400
    
    hamiltonian, drive_pulse, drive_op, DRIVE_PERIOD, method, transmon = (
        build_system
    )

    quasienergies, branches = get_branches(
        hamiltonian=hamiltonian,
        drive_pulse=drive_pulse,
        drive_op=drive_op,
        drive_period=DRIVE_PERIOD,
        method=method,
        progress_meter=True
    )

    number_op = transmon.get_number_op()
    transmon_populations = jnp.real(
        jnp.einsum("bia, ij, bja -> ba", jnp.conj(branches), number_op, branches)
    )

    actual_energies = quasienergies[:comparison_cutoff, :]
    actual_transmon_populations = transmon_populations[:comparison_cutoff, :]

    print(quasienergies[:5, :5])
    print(quasienergies.shape)

    # Load the expected results from the test data
    expected_quasienergies = load_test_data["energies"][:comparison_cutoff, :]
    expected_transmon_populations = load_test_data["transmon_populations"][:comparison_cutoff, :]

    # NOTE: Successive assertions for now, can change later
    assert np.allclose(actual_energies, expected_quasienergies)
    assert np.allclose(actual_transmon_populations, expected_transmon_populations)
