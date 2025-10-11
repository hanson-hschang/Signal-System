<div align=center>

# `Signal-System`

![Python](https://img.shields.io/badge/Python-3776AB?logo=Python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=NumPy&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-00A3E0?logo=Numba&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=PyTorch&logoColor=white)



[![package: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)
[![CI/CD: pre-commit](https://img.shields.io/badge/CI/CD-pre--commit-FAB040?logo=pre-commit)](https://pre-commit.com/)
[![syntax: pyupgrade](https://img.shields.io/badge/syntax-pyupgrade-blue?logo=pyupgrade)](https://github.com/pyupgrade/pyupgrade)
[![unit test: pytest](https://img.shields.io/badge/unit_test-pytest-0A9EDC?logo=pytest)](https://docs.pytest.org/)
[![lint & format:Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![static type: mypy](https://img.shields.io/badge/static_type-mypy-blue)](https://mypy-lang.org/)


<!-- [![CI: pre-commit](https://img.shields.io/badge/CI-pre--commit-FAB040?logo=pre-commit)](https://pre-commit.com/)
[![unit test: pytest](https://img.shields.io/badge/unit_test-pytest-0A9EDC?logo=pytest)](https://docs.pytest.org/)
[![code style: black](https://img.shields.io/badge/code_style-black-black)](https://github.com/psf/black)
[![imports: isort](https://img.shields.io/badge/imports-isort-blue?labelColor=orange)](https://pycqa.github.io/isort/)
[![static type: mypy](https://img.shields.io/badge/static_type-mypy-blue)](https://mypy-lang.org/) -->


[![Ubuntu](https://github.com/hanson-hschang/Signal-System/actions/workflows/build-ubuntu.yml/badge.svg)](https://github.com/hanson-hschang/Signal-System/actions/workflows/build-ubuntu.yml)
[![Windows](https://github.com/hanson-hschang/Signal-System/actions/workflows/build-windows.yml/badge.svg)](https://github.com/hanson-hschang/Signal-System/actions/workflows/build-windows.yml)
[![macOS](https://github.com/hanson-hschang/Signal-System/actions/workflows/build-macos.yml/badge.svg)](https://github.com/hanson-hschang/Signal-System/actions/workflows/build-macos.yml)
[![release](https://img.shields.io/github/v/release/hanson-hschang/Signal-System)](https://github.com/hanson-hschang/Signal-System/releases)
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow)](https://opensource.org/licenses/MIT)

**a Python package for `Signal-System` simulation, design, analysis, and learning**

[Installation](#-installation) • [Example](#-example)

</div>

## 📦 Installation
<!--
### Requirements
  - Python version: 3.13+
  - Additional package dependencies include: [NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html), [SciPy](https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide), [Numba](https://numba.readthedocs.io/en/stable/user/5minguide.html), [PyTorch](https://pytorch.org/docs/stable/index.html), [Matplotlib](https://matplotlib.org/stable/users/explain/quick_start.html), [H5py](https://docs.h5py.org/en/stable/), [tqdm](https://tqdm.github.io/), and [Click](https://click.palletsprojects.com/en/stable/) (detailed in `pyproject.toml`)

### Installation

Before installation, create a Python virtual environment to manage dependencies and ensure a clean installation of the **Signal & System** package.

1. Create and activate a virtual environment: (One may use your preferred way to create a virtual environment.
This tutorial uses [Anaconda](https://docs.anaconda.com/) to manage environments.)

    ```properties
    # Change directory to your <working_directory>
    cd <working_directory>

    # Create a virtual environment of name <venv>
    # with Python version 3.13
    conda create --name <venv> python=3.13

    # Activate the virtual environment
    conda activate <venv>

    # Note: Exit the virtual environment
    conda deactivate
    ``` -->

Install with pip (ensure `python >= 3.13` are installed):

```properties
# Option 1: install from PyPI (latest stable version)
pip install Signal-System

# Option 2: install from GitHub repository (latest development version)
pip install git+https://github.com/hanson-hschang/Signal-System.git
```

## 📝 Example

Please refer to [`examples`](https://github.com/hanson-hschang/Signal-System/tree/main/examples) directory and learn how to use this `Signal-System` package.
Three types of examples are provided:
  - [`system`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/system) provides various dynamic system simulations.
  - [`control`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/control) provides various control methods over dynamic systems.
  - [`estimation`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/estimation) provides various filtering and smoothing examples for different type of dynamic systems.

---

<div align="center">

![signal-system.png](signal-system.png)

</div>
