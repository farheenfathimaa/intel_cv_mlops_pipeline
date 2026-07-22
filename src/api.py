import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib

app = FastAPI(
    title="Intel Image Classification API",
    description="A FastAPI service to classify images into buildings, forest, glacier, mountain, sea, or street.",
    version="1.0.0"
)

# Class names mapping
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Model paths
CNN_MODEL_PATH = 'src/models/cnn_model.keras'
BASELINE_MODEL_PATH = 'src/models/baseline_model.joblib'

# Global variables for loaded models
cnn_model = None
baseline_model = None

@app.on_event("startup")
def load_models():
    global cnn_model, baseline_model
    # Load CNN Model
    if os.path.exists(CNN_MODEL_PATH):
        try:
            cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            print("Loaded CNN model successfully.")
        except Exception as e:
            print(f"Failed to load CNN model: {e}")
    else:
        print(f"CNN model not found at {CNN_MODEL_PATH}")

    # Load Baseline Model
    if os.path.exists(BASELINE_MODEL_PATH):
        try:
            baseline_model = joblib.load(BASELINE_MODEL_PATH)
            print("Loaded Baseline model successfully.")
        except Exception as e:
            print(f"Failed to load Baseline model: {e}")
    else:
        print(f"Baseline model not found at {BASELINE_MODEL_PATH}")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "models_loaded": {
            "cnn": cnn_model is not None,
            "baseline": baseline_model is not None
        }
    }

# Advanced feature extraction for baseline model
def extract_baseline_features(image):
    # Resize to 64x64
    image_resized = cv2.resize(image, (64, 64))
    b, g, r = cv2.split(image_resized)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    return np.array([
        np.mean(r), np.mean(g), np.mean(b),
        np.std(r), np.std(g), np.std(b),
        edge_density
    ]).reshape(1, -1)

@app.post("/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    global cnn_model
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="CNN model is not loaded/available.")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        # Preprocess image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (64, 64))
        img_batch = np.expand_dims(img_resized, axis=0)  # Add batch dimension
        
        # Predict
        predictions = cnn_model.predict(img_batch)
        score = tf.nn.softmax(predictions[0])
        predicted_class_idx = np.argmax(score)
        confidence = float(score[predicted_class_idx])
        
        return {
            "prediction": CLASSES[predicted_class_idx],
            "confidence": confidence,
            "all_probabilities": {CLASSES[i]: float(score[i]) for i in range(len(CLASSES))}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/baseline")
async def predict_baseline(file: UploadFile = File(...)):
    global baseline_model
    if baseline_model is None:
        raise HTTPException(status_code=503, detail="Baseline model is not loaded/available.")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        # Extract features
        features = extract_baseline_features(img)
        
        # Predict
        predicted_class_idx = int(baseline_model.predict(features)[0])
        
        # Probabilities
        probabilities = baseline_model.predict_proba(features)[0]
        confidence = float(probabilities[predicted_class_idx])
        
        return {
            "prediction": CLASSES[predicted_class_idx],
            "confidence": confidence,
            "all_probabilities": {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
