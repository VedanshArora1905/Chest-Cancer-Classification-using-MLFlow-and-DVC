import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(app):
    # Create logs directory if it doesn't exist
    log_dir = Path(app.config['LOG_FILE']).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file handler
    file_handler = RotatingFileHandler(
        app.config['LOG_FILE'],
        maxBytes=1024 * 1024,  # 1MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure Flask logger
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    
    # Log startup message
    app.logger.info('Application startup') 