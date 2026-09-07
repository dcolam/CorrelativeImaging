"""Shared test fixtures."""

import os

import pytest


@pytest.fixture(scope="session")
def qapp():
    """One offscreen QApplication for the whole session.

    Must be kept alive by a reference: an unassigned ``QApplication([])`` is
    garbage-collected immediately and Qt then aborts the process on the next
    widget construction. Session-scoped because Qt allows only one instance.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
