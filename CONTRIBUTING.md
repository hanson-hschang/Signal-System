<div align=center>

# Contributing to `Signal-System`

Thank you for your interest in contributing to `Signal-System`!

We welcome contributions from the community and appreciate your help in making this project better.

[How to Contribute](#-how-to-contribute) • [Developer Setup](#-developer-setup) • [Pull Request Process](#-pull-request-process) • [Recognition](#-recognition)

</div>


## 🎯 How to Contribute

### I. Reporting Issues

- **Search existing issues** first to avoid duplicates
- **Use issue templates** when available
- **Provide detailed information**:
  - Python version
  - Operating system
  - Steps to reproduce
  - Expected vs actual behavior
  - Screenshots (if applicable)

### II. Suggesting Features

- **Check existing feature requests** first
- **Describe the use case** clearly
- **Explain why** the feature would be valuable
- **Consider backwards compatibility**

### III. Code Contributions

We especially welcome contributions in these areas:

- **High Priority**
  - **Bug fixes** and stability improvements
  - **Performance optimizations**
  - **Chore tasks** like dependency updates and documentation improvements

- **Medium Priority**
  - **Code organization** and refactoring
  - **New features** that align with project goals
  - **Test coverage** enhancements

- **Low Priority**
  - **Minor enhancements** and quality-of-life improvements
  - **Examples** and tutorials

<details>

<summary>Detailed Code Contribution Steps</summary>

1. **Fork the repository** on GitHub to your own account.

2. **Clone your fork** locally:
   ```properties
   git clone https://github.com/your-username/pixelate.git
   cd pixelate
   ```

3. **Create a feature branch**:
   ```properties
   git checkout -b feat/your-feature-name
   ```

4. **Make your changes**:
   - Follow existing code style
   - Write tests for new features or bug fixes
   - Update documentation if needed

5. **Commit your changes**:
   ```properties
   git add .
   git commit -m "feat: add amazing new feature"
   ```

6. **Push to your fork**:
   ```properties
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

8. **Engage in the review process**:
   - Respond to feedback
   - Make requested changes
   - Ensure all checks pass

9. **Celebrate your contribution** once merged!

</details>

## 🚀 Developer Setup

### I. Prerequisites
This project uses [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management during development.
Install uv first:

```properties
# Install uv with pip
pip install uv
```

### II. Development Installation
Clone the repository and install with dev dependencies:
```properties
git clone https://github.com/hanson-hschang/pixelate.git
cd pixelate
uv sync --all-groups --all-extras
```

<details>

<summary>Detail of Development Installation</summary>

This installs the package in editable mode along with all development dependencies including testing, linting, formatting, and type checking tools.

**Development Tools:**
- pre-commit (git hooks)
- pyupgrade (syntax upgrades)
- pytest (testing framework)
- Ruff (linting and formatting)
- mypy (static type checking)
- bandit (security checking)

**Pre-commit Hooks:** Install pre-commit hooks to ensure code quality before commits
```properties
# Install pre-commit hooks
uv run pre-commit install
```

**Testing:** Use pytest for testing
```properties
# Run tests
uv run pytest --cov=src --cov-branch -c pyproject.toml
```

**Linting and Formatting:** Use Ruff for linting and formatting
```properties
# Check for linting issues
uv run ruff check src tests
# Automatically fix linting issues
uv run ruff format src tests
uv run ruff check src tests --fix
```

**Type Checking:** Use mypy for static type checking
```properties
# Run static type checking
uv run mypy src tests
```

</details>

### III. Code Quality
- **Linting & Formatting**: Code is checked and formatted with [Ruff](https://docs.astral.sh/ruff/)
- **Type checking**: Static type checking with [mypy](https://mypy.readthedocs.io/)
- **Security**: Security checks with [bandit](https://bandit.readthedocs.io/)

### IV. CI/CD
This project uses GitHub Actions for continuous integration with separate workflows:
- **Test workflow**: Runs tests on Ubuntu, Windows, and macOS
- **Linting workflow**: Ensures code adheres to linting standards
- **Security workflow**: CodeQL security scanning with write permissions for security results
- All checks must pass before merging to main branch


### V. Commit Message Guidelines

Use conventional commit format:

```
type(scope): short description

[optional body]

[optional footer]
```

<details>

<summary>Detailed Commit Message Guidelines</summary>

**Types**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```properties
feat(cli): add --verbose flag for detailed output
fix(parser): handle edge case for empty input
docs(readme): update installation instructions
refactor(test): simplify test setup
test(utility): add tests for new utility functions
chore(deps): update dependencies to latest versions
```

</details>

## 🔄 Pull Request Process

### I. Before Submitting

- [ ] Test changes locally
- [ ] Update documentation if needed
- [ ] Follow commit message guidelines
- [ ] Ensure backwards compatibility
- [ ] Check for console errors

### II. PR Description

Include in your PR description:

1. **What changes were made**
2. **Why the changes were necessary**
3. **How to test the changes**
4. **Screenshots** (for visual changes)
5. **Breaking changes** (if any)

### III. Review Process

- Maintainers will review PRs as time permits
- You may be asked to make changes
- Once approved, your PR will be merged
- Your contribution will be credited

## 🎉 Recognition

Contributors are recognized in several ways:

- **GitHub contributors** list
- **Changelog** mentions for significant contributions

We appreciate your time and effort in helping to improve this project.

---

<div align="center">

![signal-system.png](signal-system.png)

</div>
