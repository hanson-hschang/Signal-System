# Examples

This directory contains example implementations using the **Signal & System** package for system, control, and estimation.

## Directory Structure

```
examples
├── system
│   ├── cart_pole
│   ├── markov
│   └── mass_spring_damper
├── control
│   ├── lqg_mass_spring_damper
│   ├── mppi_cart_pole
│   └── mppi_mass_spring_damper
└── estimation
    └── filtering
        └── hmm
            ├── hmm_filtering
            └── learning_hmm_filtering
```

## Usage

First, ensure you have activated your Python 3.11 virtual environment that is installed with this **Signal & System** package.

Examples can then be run by changing to the specific directory and using Python's module execution. For example:
```properties
# For system examples
cd examples/system
python -m cart_pole
python -m markov
python -m mass_spring_damper

# For control examples
cd examples/control
python -m lqg_mass_spring_damper
python -m mppi_cart_pole
python -m mppi_mass_spring_damper

# For HMM examples
cd examples/estimation/filtering/hmm
python -m hmm_filtering
python -m learning_hmm_filtering
```

## Example Categories

### System
Implementations of different dynamical systems:
- Cart-pole system
- Markov chains
- Mass-spring-damper systems

### Control
Examples demonstrating various control strategies:
- Linear Quadratic Gaussian (LQG) control
- Model Predictive Path Integral (MPPI) control
- Applications to common mechanical systems

### Estimation
State estimation and filtering techniques:
#### HMM (Hidden Markov Models)
- Basic HMM filtering implementation
- HMM learning and filtering examples


## Adding New Examples
When adding new examples:
1. Place your example in the appropriate category folder
2. Ensure your code is well-documented with comments
3. Update this `README` if adding new categories or significant examples
