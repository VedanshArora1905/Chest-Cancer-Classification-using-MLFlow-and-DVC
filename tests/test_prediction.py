# """
# Unit tests for PredictionPipeline
# """
# import pytest
# import numpy as np
# import os
# from unittest.mock import Mock, patch, MagicMock
# from pathlib import Path
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# from ChestCancerClassifier.pipeline.prediction import PredictionPipeline


# class TestPredictionPipeline:
#     """Test cases for PredictionPipeline class"""

#     def test_init(self):
#         """Test PredictionPipeline initialization"""
#         pipeline = PredictionPipeline("test_image.jpg")
#         assert pipeline.filename == "test_image.jpg"
#         assert pipeline.class_mapping == {
#             0: 'Adenocarcinoma Cancer',
#             1: 'NORMAL'
#         }

#     def test_class_mapping_correct(self):
#         """Test that class mapping is correct (0=Cancer, 1=Normal)"""
#         pipeline = PredictionPipeline("test.jpg")
#         assert pipeline.class_mapping[0] == 'Adenocarcinoma Cancer'
#         assert pipeline.class_mapping[1] == 'NORMAL'

#     @patch('ChestCancerClassifier.pipeline.prediction.load_model')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.load_img')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.img_to_array')
#     @patch('os.path.exists')
#     def test_predict_cancer(self, mock_exists, mock_img_to_array, 
#                             mock_load_img, mock_load_model):
#         """Test prediction for cancer image (class 0)"""
#         # Setup mocks
#         mock_exists.return_value = True
        
#         # Create mock model
#         mock_model = MagicMock()
#         mock_model.predict.return_value = np.array([[0.8, 0.2]])  # 80% cancer
#         mock_load_model.return_value = mock_model
        
#         # Mock image loading
#         mock_img = MagicMock()
#         mock_load_img.return_value = mock_img
#         mock_img_array = np.random.rand(224, 224, 3)
#         mock_img_to_array.return_value = mock_img_array
        
#         # Test prediction
#         pipeline = PredictionPipeline("test_cancer.jpg")
#         result = pipeline.predict()
        
#         # Assertions
#         assert len(result) == 1
#         assert result[0]['image'] == 'Adenocarcinoma Cancer'
#         assert result[0]['class_index'] == 0
#         assert result[0]['confidence'] == 0.8
#         assert result[0]['probabilities']['adenocarcinoma'] == 0.8
#         assert result[0]['probabilities']['normal'] == 0.2
#         mock_model.predict.assert_called_once()

#     @patch('ChestCancerClassifier.pipeline.prediction.load_model')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.load_img')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.img_to_array')
#     @patch('os.path.exists')
#     def test_predict_normal(self, mock_exists, mock_img_to_array, 
#                            mock_load_img, mock_load_model):
#         """Test prediction for normal image (class 1)"""
#         # Setup mocks
#         mock_exists.return_value = True
        
#         # Create mock model
#         mock_model = MagicMock()
#         mock_model.predict.return_value = np.array([[0.1, 0.9]])  # 90% normal
#         mock_load_model.return_value = mock_model
        
#         # Mock image loading
#         mock_img = MagicMock()
#         mock_load_img.return_value = mock_img
#         mock_img_array = np.random.rand(224, 224, 3)
#         mock_img_to_array.return_value = mock_img_array
        
#         # Test prediction
#         pipeline = PredictionPipeline("test_normal.jpg")
#         result = pipeline.predict()
        
#         # Assertions
#         assert len(result) == 1
#         assert result[0]['image'] == 'NORMAL'
#         assert result[0]['class_index'] == 1
#         assert result[0]['confidence'] == 0.9
#         assert result[0]['probabilities']['adenocarcinoma'] == 0.1
#         assert result[0]['probabilities']['normal'] == 0.9

#     @patch('os.path.exists')
#     def test_predict_model_not_found(self, mock_exists):
#         """Test error handling when model file doesn't exist"""
#         mock_exists.return_value = False
        
#         pipeline = PredictionPipeline("test.jpg")
#         with pytest.raises(FileNotFoundError) as exc_info:
#             pipeline.predict()
        
#         assert "Model not found" in str(exc_info.value)

#     @patch('ChestCancerClassifier.pipeline.prediction.load_model')
#     @patch('os.path.exists')
#     def test_predict_image_not_found(self, mock_exists, mock_load_model):
#         """Test error handling when image file doesn't exist"""
#         # Model exists, but image doesn't
#         def exists_side_effect(path):
#             return "model.h5" in str(path)
        
#         mock_exists.side_effect = exists_side_effect
#         mock_load_model.return_value = MagicMock()
        
#         pipeline = PredictionPipeline("nonexistent.jpg")
#         with pytest.raises(FileNotFoundError) as exc_info:
#             pipeline.predict()
        
#         assert "Image not found" in str(exc_info.value)

#     @patch('ChestCancerClassifier.pipeline.prediction.load_model')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.load_img')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.img_to_array')
#     @patch('os.path.exists')
#     def test_image_normalization(self, mock_exists, mock_img_to_array, 
#                                  mock_load_img, mock_load_model):
#         """Test that images are normalized correctly (divided by 255)"""
#         mock_exists.return_value = True
#         mock_model = MagicMock()
#         mock_model.predict.return_value = np.array([[0.5, 0.5]])
#         mock_load_model.return_value = mock_model
        
#         # Create image array with values 0-255
#         mock_img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
#         mock_img_to_array.return_value = mock_img_array
        
#         mock_load_img.return_value = MagicMock()
        
#         pipeline = PredictionPipeline("test.jpg")
#         pipeline.predict()
        
#         # Check that predict was called with normalized image
#         call_args = mock_model.predict.call_args[0][0]
#         assert call_args.max() <= 1.0, "Image should be normalized to [0, 1]"
#         assert call_args.min() >= 0.0, "Image should be normalized to [0, 1]"

#     @patch('ChestCancerClassifier.pipeline.prediction.load_model')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.load_img')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.img_to_array')
#     @patch('os.path.exists')
#     def test_batch_dimension_added(self, mock_exists, mock_img_to_array, 
#                                    mock_load_img, mock_load_model):
#         """Test that batch dimension is added to image"""
#         mock_exists.return_value = True
#         mock_model = MagicMock()
#         mock_model.predict.return_value = np.array([[0.5, 0.5]])
#         mock_load_model.return_value = mock_model
        
#         mock_img_array = np.random.rand(224, 224, 3)
#         mock_img_to_array.return_value = mock_img_array
#         mock_load_img.return_value = MagicMock()
        
#         pipeline = PredictionPipeline("test.jpg")
#         pipeline.predict()
        
#         # Check that predict was called with batched image
#         call_args = mock_model.predict.call_args[0][0]
#         assert len(call_args.shape) == 4, "Image should have batch dimension"
#         assert call_args.shape[0] == 1, "Batch size should be 1"

#     @patch('ChestCancerClassifier.pipeline.prediction.load_model')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.load_img')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.img_to_array')
#     @patch('os.path.exists')
#     def test_prediction_output_structure(self, mock_exists, mock_img_to_array, 
#                                         mock_load_img, mock_load_model):
#         """Test that prediction output has correct structure"""
#         mock_exists.return_value = True
#         mock_model = MagicMock()
#         mock_model.predict.return_value = np.array([[0.6, 0.4]])
#         mock_load_model.return_value = mock_model
        
#         mock_img_array = np.random.rand(224, 224, 3)
#         mock_img_to_array.return_value = mock_img_array
#         mock_load_img.return_value = MagicMock()
        
#         pipeline = PredictionPipeline("test.jpg")
#         result = pipeline.predict()
        
#         # Check output structure
#         assert isinstance(result, list)
#         assert len(result) == 1
#         assert 'image' in result[0]
#         assert 'confidence' in result[0]
#         assert 'class_index' in result[0]
#         assert 'probabilities' in result[0]
#         assert 'adenocarcinoma' in result[0]['probabilities']
#         assert 'normal' in result[0]['probabilities']

#     @patch('ChestCancerClassifier.pipeline.prediction.load_model')
#     @patch('ChestCancerClassifier.pipeline.prediction.image.load_img')
#     @patch('os.path.exists')
#     def test_prediction_exception_handling(self, mock_exists, mock_load_img, 
#                                            mock_load_model):
#         """Test that exceptions are properly handled and logged"""
#         mock_exists.return_value = True
#         mock_load_model.side_effect = Exception("Model loading error")
        
#         pipeline = PredictionPipeline("test.jpg")
#         with pytest.raises(Exception) as exc_info:
#             pipeline.predict()
        
#         assert "Model loading error" in str(exc_info.value)

