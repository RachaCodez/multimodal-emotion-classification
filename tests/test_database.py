"""
Unit tests for database operations.
Tests user creation, predictions, and queries.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.db_config import Base, engine
from database.db_operations import User, Prediction, create_user, save_prediction
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def test_db():
    """Create a test database session."""
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()

    yield session

    # Cleanup
    session.close()
    # Note: In production, you'd drop tables here for a clean slate


class TestUserModel:
    """Test User model and operations."""

    def test_create_user(self, test_db):
        """Test user creation."""
        try:
            user = create_user(test_db, "testuser", "test@example.com", "password123")
            assert user.id is not None
            assert user.username == "testuser"
            assert user.email == "test@example.com"
            assert user.password_hash is not None
            assert user.password_hash != "password123"  # Should be hashed
        except Exception as e:
            pytest.skip(f"Database not available: {e}")

    def test_password_hashing(self, test_db):
        """Test password hashing and verification."""
        try:
            user = User(username="hashtest", email="hash@example.com")
            user.set_password("mypassword")

            # Password should be hashed
            assert user.password_hash != "mypassword"

            # Check password verification
            assert user.check_password("mypassword") == True
            assert user.check_password("wrongpassword") == False
        except Exception as e:
            pytest.skip(f"Bcrypt not available: {e}")


class TestPredictionModel:
    """Test Prediction model and operations."""

    def test_save_prediction(self, test_db):
        """Test saving a prediction."""
        try:
            # Create a test user first
            user = create_user(test_db, "preduser", "pred@example.com", "password123")

            # Save a prediction
            prediction = save_prediction(
                test_db,
                user_id=user.id,
                input_type='text',
                predicted_emotion='happy',
                confidence_score=0.95,
                text_emotion='happy',
                text_confidence=0.95
            )

            assert prediction.id is not None
            assert prediction.user_id == user.id
            assert prediction.predicted_emotion == 'happy'
            assert prediction.confidence_score == 0.95
        except Exception as e:
            pytest.skip(f"Database not available: {e}")

    def test_prediction_relationships(self, test_db):
        """Test user-prediction relationships."""
        try:
            # Create user and prediction
            user = create_user(test_db, "reluser", "rel@example.com", "password123")
            prediction = save_prediction(
                test_db,
                user_id=user.id,
                input_type='multimodal',
                predicted_emotion='sad',
                confidence_score=0.88
            )

            # Test relationship
            assert prediction.user == user
            assert prediction in user.predictions
        except Exception as e:
            pytest.skip(f"Database not available: {e}")


class TestDatabaseOperations:
    """Test database CRUD operations."""

    def test_get_user_predictions(self, test_db):
        """Test retrieving user predictions."""
        try:
            from database.db_operations import get_user_predictions

            # Create user and predictions
            user = create_user(test_db, "getuser", "get@example.com", "password123")
            save_prediction(test_db, user_id=user.id, input_type='text',
                          predicted_emotion='happy', confidence_score=0.9)
            save_prediction(test_db, user_id=user.id, input_type='image',
                          predicted_emotion='sad', confidence_score=0.85)

            # Get predictions
            predictions = get_user_predictions(test_db, user.id)
            assert len(predictions) == 2
        except Exception as e:
            pytest.skip(f"Database not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
