# RPA Project - Development Notes

## Project Setup

## Python

- use python 3.13 features
- specify type hints
- instead of Optional[X], List, Tuple, etc. use python 3.10+ way with X|None, list, tuple
- for data models use pydantic 2 and `Annotated`
- use Loguru `logger.debug()` with built-in templating `logger.debug("text {metavar}", metavar=expression)`. I often use a variable for both metavar and expression: `x=42; logger.debug("x={x}", x=x)`
- use loguru.debug() instead of `print()` everywhere, except for short one-off scripts.
- use pytest for testing
- use uv to manage python projects. To add dependencies use `uv add [--dev] package` without exact version. Do not edit dependencies in pyproject.toml directly.
- use ruff and mypy for static analysis
- if Makefile is available use targets: check, fix, test

### Python testing

- use pytest for testing
- use pytest plugins and fixtures even where the direct approach is available: mocking, tmp files, etc.
- try hard to avoid code-level monkey patching. If unavoidable, ask.
- whenever adding new functionality to a subpackage in that uv workspace run the tests for that package instead for the whole project.

This is a Python 3.13 project managed with `uv`, configured with strict linting (ruff) and type checking (mypy).

### Project Structure

```
rpa/
├── src/
│   └── rpa/
│       ├── __init__.py      # Package initialization
│       └── __main__.py      # Entry point for module execution
├── pyproject.toml           # Project configuration
├── Makefile                 # Build automation
└── uv.lock                  # Dependency lock file
```

### Dependencies

- **Runtime**: loguru (>=0.7.3)
- **Dev**: ruff (>=0.14.0), mypy (>=1.18.2)

### Configuration

**Ruff** (pyproject.toml:15-19):
- Line length: 120
- Target: Python 3.13
- Comprehensive linting rules enabled
- Ignored: ANN401, COM812, ISC001

**Mypy** (pyproject.toml:21-33):
- Strict type checking enabled
- Python 3.13 target
- All warnings enabled (return_any, unused_configs, etc.)

**Build System** (pyproject.toml:11-19):
- Backend: hatchling
- Source layout: src/rpa
- Entry point script: `rpa` command → `rpa.__main__:main`

## Running the Project

### Using uv

```bash
# Run as module
uv run python -m rpa

# Run using installed script
uv run rpa

# Direct Python execution
uv run python main.py
```

### Using Make

```bash
# Initialize project (create venv + sync deps)
make init

# Run linting and type checking
make check

# Auto-fix formatting and linting issues
make fix

# Format code only
make format

# Clean virtual environment
make clean
```

## Makefile Targets

- `make init`: Creates `.venv` and syncs all dependencies
- `make check`: Runs ruff check, ruff format --check, and mypy
- `make fix`: Runs ruff format and ruff check --fix
- `make clean`: Removes `.venv`
- `make help`: Lists all available targets

## Development Workflow

1. Install dependencies: `make init`
2. Make code changes in `src/rpa/`
3. Run checks: `make check`
4. Fix issues: `make fix`
5. Test execution: `uv run python -m rpa`

## Notes

- The project uses a src-layout for better package isolation
- Loguru is configured for logging (see `__main__.py`)
- The package is installed in editable mode during `uv sync`
- Entry point allows running via `uv run rpa` command
