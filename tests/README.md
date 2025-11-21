# Test Suite for Chest Cancer Classification

This directory contains comprehensive test cases for the Chest Cancer Classification project.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and configuration
├── test_prediction.py       # Unit tests for PredictionPipeline
├── test_app.py              # Integration tests for Flask app
└── test_class_mapping.py    # Tests for class mapping correctness
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src/ChestCancerClassifier --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_prediction.py
```

### Run specific test
```bash
pytest tests/test_prediction.py::TestPredictionPipeline::test_predict_cancer
```

### Run with verbose output
```bash
pytest -v
```

## Test Categories

### Unit Tests (`test_prediction.py`)
- PredictionPipeline initialization
- Class mapping correctness
- Image preprocessing (normalization, batch dimension)
- Error handling (missing files, exceptions)
- Output structure validation

### Integration Tests (`test_app.py`)
- Flask endpoints (health, home, predict, train)
- Request/response handling
- Error handling
- ClientApp class functionality

### Class Mapping Tests (`test_class_mapping.py`)
- Verifies class mapping matches training data
- Ensures 0 = Cancer, 1 = Normal
- Validates mapping structure

## Test Coverage

The test suite covers:
- ✅ Prediction pipeline functionality
- ✅ Class mapping correctness
- ✅ Image preprocessing
- ✅ Error handling
- ✅ Flask API endpoints
- ✅ Request validation
- ✅ Response structure

## Fixtures

Common fixtures are defined in `conftest.py`:
- `sample_image_path`: Creates dummy image file
- `mock_model`: Mock TensorFlow model
- `mock_model_predictions`: Mock prediction outputs
- `sample_base64_image`: Base64 encoded test image
- `app_client`: Flask test client

## Notes

- Tests use mocking to avoid loading actual models/images
- Some tests require the model to be trained first
- Integration tests may need the Flask app to be configured

