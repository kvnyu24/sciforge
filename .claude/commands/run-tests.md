# Run Tests

Run the test suite for the SciForge project.

## Usage

```
/run-tests [options]
```

## Options

- `--coverage` - Run with coverage reporting
- `--verbose` - Show detailed output
- `--module <name>` - Test specific module (physics, numerical, stochastic, etc.)

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sciforge --cov-report=term-missing

# Run specific module tests
pytest tests/test_physics/ -v
pytest tests/test_numerical/ -v

# Run single test file
pytest tests/test_physics/test_mechanics.py -v

# Run tests matching pattern
pytest tests/ -k "test_particle" -v
```

## Expected Output

Tests should pass with no failures. Coverage target is 80%.
