import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
import joblib
import tensorflow as tf

from src.data_pipeline.preprocess import extract_features_from_image
from src.api import app, extract_baseline_features

# Create a FastAPI test client as a fixture to correctly trigger startup lifespan events
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# Helper fixture to create a temporary test image
@pytest.fixture
def temp_image_path(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a white rectangle to have edges
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    
    img_file = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_file), img)
    return str(img_file)

def test_feature_extraction(temp_image_path):
    """Test that manual features are extracted correctly and have expected length."""
    features = extract_features_from_image(temp_image_path)
    assert features is not None
    # 7 features: mean R, G, B, std R, G, B, edge density
    assert len(features) == 7
    # Edge density should be greater than 0 due to the white rectangle
    assert features[6] > 0

def test_baseline_model_inference():
    """Test loading the baseline model and running prediction on a dummy feature set."""
    baseline_model_path = 'src/models/baseline_model.joblib'
    if not os.path.exists(baseline_model_path):
        pytest.skip("Baseline model not trained yet. Skipping test.")
        
    model = joblib.load(baseline_model_path)
    dummy_features = np.random.rand(1, 7)
    prediction = model.predict(dummy_features)
    assert len(prediction) == 1
    assert 0 <= prediction[0] < 6

def test_cnn_model_inference(temp_image_path):
    """Test loading the CNN model and running prediction on an image."""
    cnn_model_path = 'src/models/cnn_model.keras'
    if not os.path.exists(cnn_model_path):
        pytest.skip("CNN model not trained yet. Skipping test.")
        
    model = tf.keras.models.load_model(cnn_model_path)
    # Prepare image
    img = cv2.imread(temp_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (64, 64))
    img_batch = np.expand_dims(img_resized, axis=0)
    
    predictions = model.predict(img_batch)
    assert predictions.shape == (1, 6)

def test_api_health(client):
    """Test API health endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "models_loaded" in data
    # Both models should be successfully loaded
    assert data["models_loaded"]["cnn"] is True
    assert data["models_loaded"]["baseline"] is True

def test_api_predict_baseline(client, temp_image_path):
    """Test API predict endpoint using baseline model."""
    if not os.path.exists('src/models/baseline_model.joblib'):
        pytest.skip("Baseline model not available.")
        
    with open(temp_image_path, "rb") as f:
        response = client.post("/predict/baseline", files={"file": f})
        
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "all_probabilities" in data
    assert data["prediction"] in ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def test_api_predict_cnn(client, temp_image_path):
    """Test API predict endpoint using CNN model."""
    if not os.path.exists('src/models/cnn_model.keras'):
        pytest.skip("CNN model not available.")
        
    with open(temp_image_path, "rb") as f:
        response = client.post("/predict/cnn", files={"file": f})
        
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "all_probabilities" in data
    assert data["prediction"] in ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
