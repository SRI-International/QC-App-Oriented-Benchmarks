"""Pytest configuration for hydrogen_lattice tests."""

def pytest_addoption(parser):
    parser.addoption("--noplot", action="store_true", default=False,
                     help="Suppress matplotlib plot windows during testing")

def pytest_configure(config):
    if config.getoption("--noplot"):
        import matplotlib
        matplotlib.use("Agg")
