"""
WSGI entry point for production deployment with Gunicorn.

Usage:
    gunicorn -c gunicorn_config.py wsgi:app
"""

from app import app

if __name__ == '__main__':
    app.run()
