import os
from datetime import timedelta


class Config:
    # Secret key
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'change-this-secret-key'

    # CSRF Protection
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None  # No timeout

    # Session Security
    SESSION_COOKIE_SECURE = os.environ.get('FLASK_ENV') == 'production'  # HTTPS only in production
    SESSION_COOKIE_HTTPONLY = True  # Prevent XSS access to session cookie
    SESSION_COOKIE_SAMESITE = 'Lax'  # CSRF protection
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)  # Session expires after 24 hours
    SESSION_REFRESH_EACH_REQUEST = True  # Refresh session on each request

    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    }

    # Database (SQLite for local development)
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'database', 'emotion.db')
    
    # Full URI (use SQLite by default, override with DATABASE_URL env var)
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get('DATABASE_URL')
        or f"sqlite:///{DATABASE_PATH}"
    )


    # Model paths
    SPEECH_MODEL_PATH = os.environ.get('SPEECH_MODEL_PATH', 'models/speech_model.h5')
    SPEECH_SCALER_PATH = os.environ.get('SPEECH_SCALER_PATH', 'models/speech_scaler.pkl')
    TEXT_MODEL_PATH = os.environ.get('TEXT_MODEL_PATH', 'models/text_model.h5')
    IMAGE_MODEL_PATH = os.environ.get('IMAGE_MODEL_PATH', 'models/image_model.h5')
    FUSION_MODEL_PATH = os.environ.get('FUSION_MODEL_PATH', 'models/fusion_model.pkl')
    BERT_MODEL_PATH = os.environ.get('BERT_MODEL_PATH', 'models/bert_model')

    # Upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg'}
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # Labels
    EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
    NUM_EMOTIONS = 7

    # Audio settings
    SAMPLE_RATE = 22050
    AUDIO_DURATION = 3
    N_MFCC = 40

    # Text settings
    MAX_TEXT_LENGTH = 128

    # Image settings
    IMAGE_SIZE = (224, 224)
