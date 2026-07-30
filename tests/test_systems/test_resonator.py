from pathlib import Path

import jax
import numpy as np
import pytest

from src.blueprint.systems import Resonator
from tests.test_systems.test_systems import BaseTestSystem


class TestResonator(BaseTestSystem):
    @pytest.fixture
    def load_system(self):
        jax.config.update("jax_enable_x64", True)

        filename = "resonator_testdata.npz"
        DATA_DIR = Path.cwd() / "tests/test_systems/data"
        test_data = np.load(DATA_DIR / filename)

        frequency = test_data["RES_FREQUENCY"]
        impedance = test_data["RES_IMPEDANCE"]
        dim = int(test_data["RES_DIM"])

        resonator = Resonator.from_frequency(
            label="resonator",
            frequency=frequency,
            impedance=impedance,
            dim=dim,
        )
        return resonator, test_data
