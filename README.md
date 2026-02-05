
# blueprint

A lightweight Python library for modelling and analyzing quantum circuits and superconducting devices written with JAX and designed for GPU acceleration. Note that this project is still in an alpha version - there are features not yet supported and possible code issues. We are not promising any support with issues you may be experiencing or features that you may want to implement. However, we still encourage you to report any issues, which we can look into and potentially solve.

**Overview**

`blueprint` provides building blocks for defining quantum devices (transmons, resonators, fluxonium, Kerr oscillators), drives and pulses, couplings, and tools for computing representations and gate metrics. The project is organized as a small, extensible package suitable for simulation workflows, analysis notebooks, and integration into larger control and optimization toolchains.

**Key Features**

- Modular device models: transmon, resonator, fluxonium, kerr oscillator.
- Operator and basis utilities for building Hamiltonians and computing spectra.
- Drive and pulse primitives for time-dependent simulations.
- Gate representations and metrics for characterizing gate performance.

**Installation**

Install from source for development (we recommend using uv, but pip should work as well):

```bash
uv sync
```
or alternatively

```bash
python -m pip install .
```

Recommended: create a virtual environment before installing.

You can optional install some optional libraries for running notebooks (ipykernel) and plotting (matplotlib) with poetry by providing the optional extra arguments `notebook` and `plot`, respectively. For example:

```bash
uv sync --extra notebook --extra plot
```

Finally, if you're running this on a environment where CUDA is installed and available, you can also install the corresponding JAX dependencies with poetry by providing an extra argument as follows

```bash
uv sync --extra cuda
```

**Quick Start**

Basic usage pattern:

```python
from blueprint import systems, device

# Build a device (example, API is illustrative)
# system = systems.transmon.Transmon(...)

# Compute energies, build Hamiltonians, or run simulations
# energies = system.eigenvalues()
# print(energies[:6])
```

See the `tutorial/` folder for runnable examples and exploratory analyses.

**Package Structure**

- `blueprint/bases` — basis definitions and utilities. Useful for performing process tomography in simulation.
- `blueprint/branches` — analysis branches (Floquet, quantum, ...).
- `blueprint/couplings` — coupling models and helper routines.
- `blueprint/device` — device construction and helpers.
- `blueprint/drives` — drive, envelope, and pulse definitions.
- `blueprint/gates` — gate metrics and representations.
- `blueprint/operators` — operator primitives (charge, harmonic, ...).
- `blueprint/systems` — concrete physical system models (transmon, fluxonium, resonator).
- `tutorial/` — example notebooks demonstrating workflows.

**Development**

- Run tests with your preferred test runner (the project is configured for pytest). Test still need to be added (of course).
- Format and lint using black and ruff.

**Contributing**

Contributions are welcome. Please open issues for bugs and feature requests, and submit pull requests for proposed changes. Include tests and update the notebooks or examples when adding functionality.

**License**

This project includes a `LICENSE` file at the repository root. Refer to it for license details.
