<div align=center>
  <h1>Signal & System</h1>

![Python](https://img.shields.io/badge/Python-3776AB?logo=Python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=NumPy&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-00A3E0?logo=Numba&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=PyTorch&logoColor=white)


[![CI: pre-commit](https://img.shields.io/badge/CI-pre--commit-FAB040?logo=pre-commit)](https://pre-commit.com/)
[![unit test: pytest](https://img.shields.io/badge/unit_test-pytest-0A9EDC?logo=pytest)](https://docs.pytest.org/)
[![code style: black](https://img.shields.io/badge/code_style-black-black)](https://github.com/psf/black)
[![imports: isort](https://img.shields.io/badge/imports-isort-blue?labelColor=orange)](https://pycqa.github.io/isort/)
[![static type: mypy](https://img.shields.io/badge/static_type-mypy-blue)](https://mypy-lang.org/)

[![build](https://github.com/hanson-hschang/Signal-System/actions/workflows/build.yml/badge.svg)](https://github.com/hanson-hschang/Signal-System/actions)
[![release](https://img.shields.io/github/v/release/hanson-hschang/Signal-System)](https://github.com/hanson-hschang/Signal-System/releases)
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow)](https://opensource.org/licenses/MIT)

</div>

A Python package for **Signal & System** simulation, design, analysis, and learning.

## Dependency & installation

### Requirements
  - Python version: 3.11
  - Additional package dependencies include: [NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html), [SciPy](https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide), [Numba](https://numba.readthedocs.io/en/stable/user/5minguide.html), [PyTorch](https://pytorch.org/docs/stable/index.html), [Matplotlib](https://matplotlib.org/stable/users/explain/quick_start.html), [H5py](https://docs.h5py.org/en/stable/), [tqdm](https://tqdm.github.io/), and [Click](https://click.palletsprojects.com/en/stable/) (detailed in `pyproject.toml`)

### Installation

Before installation, create a Python virtual environment to manage dependencies and ensure a clean installation of the **Signal & System** package.

1. Create and activate a virtual environment: (One may use your preferred way to create a virtual environment. This tutorial uses [Anaconda](https://docs.anaconda.com/) to manage the environments.)

    ```properties
    # Change directory to your working folder
    cd path_to_your_working_folder

    # Create a virtual environment of name `myenv`
    # with Python version 3.11
    conda create --name myenv python=3.11

    # Activate the virtual environment
    conda activate myenv

    # Note: Exit the virtual environment
    conda deactivate
    ```

2. Install Package: (two methods)

    ```properties
    # Install directly from GitHub
    pip install git+https://github.com/hanson-hschang/Signal-System.git

    # Or clone and install
    git clone https://github.com/hanson-hschang/Signal-System.git
    cd Signal-System
    pip install .
    ```

<details>

<summary> Click me to expand/collapse developer environment setup </summary>

## Developer environment setup

1. Clone and install development dependencies:
    ```properties
    git clone https://github.com/hanson-hschang/Signal-System.git
    cd Signal-System
    pip install -e ".[dev]"
    ```

2. Generate development requirements file:
    ```properties
    pip-compile pyproject.toml --extra=dev --output-file=requirements-dev.txt
    ```

3. Configure pre-commit hooks:
    ```properties
    pre-commit install
    ```

### Development Tools

This project uses several tools for quality assurance:

- [pre-commit](https://pre-commit.com/): Git hooks for code quality checks
- [pytest](https://docs.pytest.org/en/stable/): Unit testing
- [Black](https://black.readthedocs.io/en/stable/): Code formatting
- [isort](https://pycqa.github.io/isort/): Package import sorting
- [mypy](https://mypy.readthedocs.io/en/stable/): Static type checking

### Running Tests
```properties
# Standard test execution
pytest -c pyproject.toml

# Run tests with coverage report
pytest -c pyproject.toml --cov=src --cov-report=xml --cov-report=term
```

### Code Style Guidelines

- Adherence to [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Mandatory type hints for all functions and variables
- Documentation using  [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format specification

Format codebase:
```properties
# Upgrade Python syntax
pyupgrade --exit-zero-even-if-changed --py38-plus src/**/*.py

# Apply code formatting
black --config pyproject.toml ./

# Perform static type checking
mypy --config-file pyproject.toml ./

# Organize imports
isort --settings-path pyproject.toml ./
```

</details>

## Example

Please refer to [`examples`](https://github.com/hanson-hschang/Signal-System/tree/main/examples) directory and learn how to use this **Signal & System** package.
Three types of examples are provided:
  - [`system`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/system) provides various dynamic system simulations.
  - [`control`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/control) provides various control methods over dynamic systems.
  - [`estimation`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/estimation) provides various filtering and smoothing examples for different type of dynamic systems.

## License

This project is released under the [MIT License](https://github.com/hanson-hschang/Signal-System/blob/main/LICENSE).

## Contributing

1. Fork this repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest -c pyproject.toml`)
5. Commit your changes (`git commit -m "feat: Add some amazing feature"`)
6. Push to the feature branch (`git push origin feat/amazing-feature`)
7. Open a Pull Request
