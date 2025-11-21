# """
# Pytest configuration and fixtures for testing
# """
# import pytest
# import os
# import numpy as np
# from pathlib import Path
# from unittest.mock import Mock, patch, MagicMock
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # Test directories
# TEST_DIR = Path(__file__).parent
# PROJECT_ROOT = TEST_DIR.parent


# @pytest.fixture
# def sample_image_path(tmp_path):
#     """Create a dummy image file for testing"""
#     img_path = tmp_path / "test_image.jpg"
#     # Create a minimal valid image file (1x1 pixel PNG)
#     # In real tests, you'd use actual image files
#     img_path.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82')
#     return str(img_path)


# @pytest.fixture
# def mock_model():
#     """Create a mock TensorFlow model for testing"""
#     model = Sequential([
#         Dense(2, activation='softmax', input_shape=(224, 224, 3))
#     ])
#     return model


# @pytest.fixture
# def mock_model_predictions():
#     """Mock model predictions"""
#     return np.array([[0.7, 0.3]])  # 70% cancer, 30% normal


# @pytest.fixture
# def sample_base64_image():
#     """Sample base64 encoded image string"""
#     # Minimal valid base64 image (1x1 pixel PNG)
#     return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


# @pytest.fixture
# def model_path(tmp_path):
#     """Create a temporary model path"""
#     model_dir = tmp_path / "artifacts" / "training"
#     model_dir.mkdir(parents=True, exist_ok=True)
#     return str(model_dir / "model.h5")


# @pytest.fixture
# def app_client():
#     """Create a Flask test client"""
#     from app import app
#     app.config['TESTING'] = True
#     with app.test_client() as client:
#         yield client

