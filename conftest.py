"""Register the one custom pytest marker this suite uses.

The former `sys.path` manipulation is gone: run the suite as
`python -m pytest` from the project root, which puts the root on `sys.path`
without any help from us.

What remains is not a path hack. Without this registration pytest emits a
`PytestUnknownMarkWarning` on every run, which teaches readers to ignore
warnings, and the marker's meaning would be documented nowhere.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running integration test (tiny-overfit); run with -m slow"
    )
