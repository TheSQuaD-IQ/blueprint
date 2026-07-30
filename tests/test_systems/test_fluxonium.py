from pathlib import Path

import jax
import numpy as np
import pytest

from src.blueprint.systems import Fluxonium
from tests.test_systems.test_systems import BaseTestSystem


class TestFluxonium(BaseTestSystem):
    @pytest.fixture
    def load_system(self):
        jax.config.update("jax_enable_x64", True)

        filename = "fluxonium_testdata.npz"
        DATA_DIR = Path.cwd() / "tests/test_systems/data"
        test_data = np.load(DATA_DIR / filename)

        josephson_energy = test_data["JOSEPHSON_ENERGY"]
        inductive_energy = test_data["INDUCTIVE_ENERGY"]
        charging_energy = test_data["CHARGING_ENERGY"]
        harmonic_cutoff = int(test_data["HARMONIC_CUTOFF"])
        dim = int(test_data["FLUXONIUM_DIM"])
        ext_flux = test_data["EXT_FLUX"]

        fluxonium = Fluxonium(
            label="fluxonium",
            charging_energy=charging_energy,
            josephson_energy=josephson_energy,
            inductive_energy=inductive_energy,
            harmonic_cutoff=harmonic_cutoff,
            ext_flux=ext_flux,
            dim=dim,
        )
        return fluxonium, test_data
