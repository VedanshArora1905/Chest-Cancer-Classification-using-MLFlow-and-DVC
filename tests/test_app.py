# """
# Integration tests for Flask application
# """
# import pytest
# import json
# import base64
# from unittest.mock import patch, MagicMock
# from pathlib import Path

# from app import app, ClientApp


# @pytest.fixture
# def client():
#     """Create a Flask test client"""
#     app.config['TESTING'] = True
#     with app.test_client() as client:
#         yield client


# @pytest.fixture
# def sample_base64_image():
#     """Sample base64 encoded image"""
#     # Minimal valid base64 image (1x1 pixel PNG)
#     return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


# class TestHealthEndpoint:
#     """Test cases for health check endpoint"""

#     def test_health_check(self, client):
#         """Test health check endpoint"""
#         response = client.get('/health')
#         assert response.status_code == 200
#         data = json.loads(response.data)
#         assert data['status'] == 'healthy'


# class TestHomeEndpoint:
#     """Test cases for home endpoint"""

#     def test_home_route(self, client):
#         """Test home route returns HTML"""
#         response = client.get('/')
#         assert response.status_code == 200
#         assert response.content_type == 'text/html; charset=utf-8'


# class TestPredictEndpoint:
#     """Test cases for prediction endpoint"""

#     def test_predict_missing_json(self, client):
#         """Test prediction endpoint with non-JSON request"""
#         response = client.post('/predict', data='not json')
#         assert response.status_code == 400
#         data = json.loads(response.data)
#         assert 'error' in data
#         assert 'JSON' in data['error']

#     def test_predict_missing_image(self, client):
#         """Test prediction endpoint without image data"""
#         response = client.post(
#             '/predict',
#             json={},
#             content_type='application/json'
#         )
#         assert response.status_code == 400
#         data = json.loads(response.data)
#         assert 'error' in data
#         assert 'image' in data['error'].lower()

#     @patch('app.ClientApp')
#     def test_predict_success_cancer(self, mock_client_app, client, sample_base64_image):
#         """Test successful prediction for cancer image"""
#         # Mock the prediction result
#         mock_result = [{
#             "image": "Adenocarcinoma Cancer",
#             "confidence": 0.85,
#             "class_index": 0,
#             "probabilities": {
#                 "adenocarcinoma": 0.85,
#                 "normal": 0.15
#             }
#         }]
        
#         mock_app_instance = MagicMock()
#         mock_app_instance.predict.return_value = mock_result
#         mock_client_app.return_value = mock_app_instance
        
#         # Need to reinitialize the app's clApp
#         with patch('app.clApp', mock_app_instance):
#             response = client.post(
#                 '/predict',
#                 json={'image': sample_base64_image},
#                 content_type='application/json'
#             )
        
#         assert response.status_code == 200
#         data = json.loads(response.data)
#         assert len(data) == 1
#         assert data[0]['image'] == 'Adenocarcinoma Cancer'
#         assert data[0]['class_index'] == 0
#         assert data[0]['confidence'] == 0.85

#     @patch('app.ClientApp')
#     def test_predict_success_normal(self, mock_client_app, client, sample_base64_image):
#         """Test successful prediction for normal image"""
#         # Mock the prediction result
#         mock_result = [{
#             "image": "NORMAL",
#             "confidence": 0.92,
#             "class_index": 1,
#             "probabilities": {
#                 "adenocarcinoma": 0.08,
#                 "normal": 0.92
#             }
#         }]
        
#         mock_app_instance = MagicMock()
#         mock_app_instance.predict.return_value = mock_result
#         mock_client_app.return_value = mock_app_instance
        
#         with patch('app.clApp', mock_app_instance):
#             response = client.post(
#                 '/predict',
#                 json={'image': sample_base64_image},
#                 content_type='application/json'
#             )
        
#         assert response.status_code == 200
#         data = json.loads(response.data)
#         assert data[0]['image'] == 'NORMAL'
#         assert data[0]['class_index'] == 1

#     @patch('app.ClientApp')
#     def test_predict_error_handling(self, mock_client_app, client, sample_base64_image):
#         """Test error handling in prediction endpoint"""
#         mock_app_instance = MagicMock()
#         mock_app_instance.predict.side_effect = Exception("Prediction failed")
#         mock_client_app.return_value = mock_app_instance
        
#         with patch('app.clApp', mock_app_instance):
#             response = client.post(
#                 '/predict',
#                 json={'image': sample_base64_image},
#                 content_type='application/json'
#             )
        
#         assert response.status_code == 500
#         data = json.loads(response.data)
#         assert 'error' in data


# class TestClientApp:
#     """Test cases for ClientApp class"""

#     @patch('app.PredictionPipeline')
#     @patch('app.decodeImage')
#     def test_client_app_predict(self, mock_decode, mock_pipeline_class):
#         """Test ClientApp predict method"""
#         # Setup mocks
#         mock_pipeline = MagicMock()
#         mock_pipeline.predict.return_value = [{"image": "NORMAL"}]
#         mock_pipeline_class.return_value = mock_pipeline
        
#         client_app = ClientApp()
#         result = client_app.predict("base64_image_data")
        
#         # Verify calls
#         mock_decode.assert_called_once_with("base64_image_data", "inputImage.jpg")
#         mock_pipeline.predict.assert_called_once()
#         assert result == [{"image": "NORMAL"}]

#     @patch('app.PredictionPipeline')
#     @patch('app.decodeImage')
#     def test_client_app_predict_error(self, mock_decode, mock_pipeline_class):
#         """Test ClientApp error handling"""
#         mock_decode.side_effect = Exception("Decode error")
#         mock_pipeline = MagicMock()
#         mock_pipeline_class.return_value = mock_pipeline
        
#         client_app = ClientApp()
#         with pytest.raises(Exception):
#             client_app.predict("base64_image_data")


# class TestTrainEndpoint:
#     """Test cases for training endpoint"""

#     @patch('app.subprocess.run')
#     def test_train_success(self, mock_subprocess, client):
#         """Test successful training"""
#         mock_subprocess.return_value = MagicMock(returncode=0, stderr="")
        
#         response = client.post('/train')
#         assert response.status_code == 200
#         data = json.loads(response.data)
#         assert 'message' in data
#         assert 'success' in data['message'].lower()

#     @patch('app.subprocess.run')
#     def test_train_failure(self, mock_subprocess, client):
#         """Test training failure"""
#         mock_subprocess.return_value = MagicMock(
#             returncode=1, 
#             stderr="Training error"
#         )
        
#         response = client.post('/train')
#         assert response.status_code == 500
#         data = json.loads(response.data)
#         assert 'error' in data

#     @patch('app.subprocess.run')
#     def test_train_exception(self, mock_subprocess, client):
#         """Test training exception handling"""
#         mock_subprocess.side_effect = Exception("Subprocess error")
        
#         response = client.post('/train')
#         assert response.status_code == 500
#         data = json.loads(response.data)
#         assert 'error' in data

