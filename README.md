<div align=center>
  <h1>Signal & System</h1>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>
<a href='https://github.com/hanson-hschang/Signal-System/actions'>
    <img src='https://github.com/hanson-hschang/Signal-System/actions/workflows/main.yml/badge.svg' alt='CI' />
</a>
</div>

A Python package for **Signal & System** simulation, design, analysis, and learning.

## Dependency & installation

### Requirements
  - Python version: 3.11
  - Additional package dependencies include: NumPy, SciPy, Numba, PyTorch, Matplotlib, H5py, tqdm, and Click (detailed in `pyproject.toml`)

### Installation

```bash
git clone https://github.com/hanson-hschang/Signal-System.git
cd Signal-System
pip install .
```

## Example

Please refer to [`examples` directory](https://github.com/hanson-hschang/Signal-System/tree/main/examples) and learn how to use this **Signal & System** library.
Three types of examples are provided:
  - [`system`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/system) provides various dynamic system simulations.
  - [`control`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/control) provides various control methods over dynamic systems.
  - [`estimation`](https://github.com/hanson-hschang/Signal-System/tree/main/examples/estimation) provides various filtering and smoothing examples for different type of dynamic systems.

## Developer environment setup

1. Install development dependencies:
```bash
git clone https://github.com/hanson-hschang/Signal-System.git
cd Signal-System
pip install -e ".[dev]"
```

2. Generate `requirements-dev.txt` including development dependencies
```bash
pip-compile pyproject.toml --extra=dev --output-file=requirements-dev.txt
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

### Development Tools

The project uses several tools for quality assurance:

- **pre-commit**: Git hooks for code quality checks
- **pytest**: Unit testing
- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Static type checking

### Running Tests

```bash
pytest -c pyproject.toml
```

Run tests with coverage report:
```bash
pytest -c pyproject.toml --cov=src --cov-report=xml --cov-report=term
```

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Type hints are required for all functions
- Documentation strings should follow Google style

Format codebase:
```bash
# Upgrade Python syntax
pyupgrade --exit-zero-even-if-changed --py38-plus src/**/*.py

# Sort imports
isort --settings-path pyproject.toml ./

# Format code
black --config pyproject.toml ./

# Type checking
mypy --config-file pyproject.toml ./
```


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest tests/`)
5. Commit your changes (`git commit -m "feat: Add some amazing feature"`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request
