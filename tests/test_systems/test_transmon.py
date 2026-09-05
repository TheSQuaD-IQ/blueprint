from pathlib import Path

import jax
import numpy as np
import pytest

from blueprint.systems import Transmon
from tests.test_systems.test_systems import BaseTestSystem


class TestTransmon(BaseTestSystem):
    @pytest.fixture
    def load_system(self):
        jax.config.update("jax_enable_x64", True)

        filename = "transmon_testdata.npz"
        DATA_DIR = Path.cwd() / "tests/test_systems/data"
        test_data = np.load(DATA_DIR / filename)

        josephson_energy = test_data["JOSEPHSON_ENERGY"]
        charging_energy = test_data["CHARGING_ENERGY"]
        offset_charge = test_data["OFFSET_CHARGE"]
        charge_cutoff = int(test_data["CHARGE_CUTOFF"])
        dim = int(test_data["TRANSMON_DIM"])

        transmon = Transmon(
            label="transmon",
            charging_energy=charging_energy,
            josephson_energy=josephson_energy,
            offset_charge=offset_charge,
            charge_cutoff=charge_cutoff,
            dim=dim,
        )
        return transmon, test_data
