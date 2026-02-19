"""
Logging configuration for production deployment.
Configures rotating file handlers and formatters.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(app):
    """
    Configure logging for the Flask application.

    Creates rotating log files in logs/ directory with:
    - Max file size: 10MB
    - Backup count: 10 files
    - Log level: INFO for production
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.mkdir('logs')

    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/emotion_classifier.log',
        maxBytes=10240000,  # 10MB
        backupCount=10
    )

    # Formatter with timestamp, level, and location
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))

    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info('Emotion Classifier startup')

    return app.logger
