<div align=center>
  <h1>Signal & System</h1>

![Python](https://img.shields.io/badge/Python-3776AB?logo=Python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=NumPy&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-00A3E0?logo=Numba&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=PyTorch&logoColor=white)

[![CI](https://github.com/hanson-hschang/Signal-System/actions/workflows/main.yml/badge.svg)](https://github.com/hanson-hschang/Signal-System/actions)
[![unit test: pytest](https://img.shields.io/badge/unit_test-pytest-blue)](https://docs.pytest.org/)
[![code style: black](https://img.shields.io/badge/code_style-black-black)](https://github.com/psf/black)
[![imports: isort](https://img.shields.io/badge/imports-isort-blue?labelColor=orange)](https://pycqa.github.io/isort/)
[![static type: mypy](https://img.shields.io/badge/static_type-mypy-blue)](https://mypy-lang.org/)


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

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

## Example

Please refer to [`examples`](https://github.com/hanson-hschang/Signal-System/tree/main/examples) directory and learn how to use this **Signal & System** package.
Three types of examples are provided:
  - [`system`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/system) provides various dynamic system simulations.
  - [`control`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/control) provides various control methods over dynamic systems.
  - [`estimation`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/estimation) provides various filtering and smoothing examples for different type of dynamic systems.

## Developer environment setup

1. Install development dependencies:
```properties
git clone https://github.com/hanson-hschang/Signal-System.git
cd Signal-System
pip install -e ".[dev]"
```

2. Generate `requirements-dev.txt` including development dependencies
```properties
pip-compile pyproject.toml --extra=dev --output-file=requirements-dev.txt
```

3. Set up pre-commit hooks:
```properties
pre-commit install
```

### Development Tools

The project uses several tools for quality assurance:

- [pre-commit](https://pre-commit.com/): Git hooks for code quality checks
- [pytest](https://docs.pytest.org/en/stable/): Unit testing
- [Black](https://black.readthedocs.io/en/stable/): Code formatting
- [isort](https://pycqa.github.io/isort/): Package import sorting
- [mypy](https://mypy.readthedocs.io/en/stable/): Static type checking

### Running Tests

```properties
pytest -c pyproject.toml
```

Run tests with coverage report:
```properties
pytest -c pyproject.toml --cov=src --cov-report=xml --cov-report=term
```

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Type hints are required for all functions
- Documentation strings should follow [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style

Format codebase:
```properties
# Upgrade Python syntax
pyupgrade --exit-zero-even-if-changed --py38-plus src/**/*.py

# Format code
black --config pyproject.toml ./

# Check static type
mypy --config-file pyproject.toml ./

# Sort imports
isort --settings-path pyproject.toml ./
```


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest -c pyproject.toml`)
5. Commit your changes (`git commit -m "feat: Add some amazing feature"`)
6. Push to the branch (`git push origin feat/amazing-feature`)
7. Open a Pull Request
