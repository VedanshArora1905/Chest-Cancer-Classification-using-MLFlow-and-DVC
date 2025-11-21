import os
from pathlib import Path

class Config:
    # Base configuration
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Security configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS configuration
    CORS_ORIGINS = [
        'http://localhost:8080',
        'http://127.0.0.1:8080',
        'https://your-production-domain.com'  # Add your production domain
    ]
    
    # MLFlow configuration
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'chest-cancer-classification')
    
    # Model configuration
    MODEL_PATH = os.getenv('MODEL_PATH', BASE_DIR / 'artifacts' / 'model')
    INPUT_SIZE = (224, 224)
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / 'logs' / 'app.log'
    
    # Rate limiting
    RATELIMIT_DEFAULT = "200 per day;50 per hour"
    RATELIMIT_STORAGE_URL = "memory://"

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
    HOST = '127.0.0.1'
    PORT = 8080
    SESSION_COOKIE_SECURE = False  # Allow HTTP in development

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    HOST = '0.0.0.0'
    PORT = int(os.getenv('PORT', 8080))
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production

class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    HOST = '127.0.0.1'
    PORT = 5000
    SESSION_COOKIE_SECURE = False
    TESTING = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Get configuration based on environment
def get_config():
    env = os.getenv('FLASK_ENV', 'default')
    return config[env] 