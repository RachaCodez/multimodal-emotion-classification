"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope='session')
def app():
    """Create application for testing."""
    try:
        from app import app as flask_app
        flask_app.config['TESTING'] = True
        flask_app.config['WTF_CSRF_ENABLED'] = False
        flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'  # Use in-memory DB for tests
        return flask_app
    except ImportError:
        pytest.skip("Application not available")


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create test CLI runner."""
    return app.test_cli_runner()
