from pathlib import Path

import jax
import numpy as np
import pytest

from blueprint.systems.resonator import Resonator
from blueprint.systems import KerrOscillator
from tests.test_systems.test_systems import BaseTestSystem


class TestKerrOscillator(BaseTestSystem):
    @pytest.fixture
    def load_system(self):
        jax.config.update("jax_enable_x64", True)

        filename = "kerr_oscillator_testdata.npz"
        DATA_DIR = Path.cwd() / "tests/test_systems/data"
        test_data = np.load(DATA_DIR / filename)

        josephson_energy = test_data["JOSEPHSON_ENERGY"]
        charging_energy = test_data["CHARGING_ENERGY"]
        dim = int(test_data["KERR_OSC_DIM"])

        kerr_oscillator = KerrOscillator(
            label="kerr_oscillator",
            josephson_energy=josephson_energy,
            charging_energy=charging_energy,
            dim=dim,
        )
        return kerr_oscillator, test_data

