# Intel Image Classification MLOps Pipeline

This repository contains a complete, production-ready Computer Vision MLOps pipeline for the Intel Image Classification dataset. The pipeline demonstrates how to reconstruct datasets from multi-part archives, perform advanced feature engineering, train baseline and deep learning models, expose them via a FastAPI model-serving API, and validate the code with automated tests.

---

## 📁 Repository Structure

```text
├── data_archive/         # Split dataset parts (git-tracked to stay under 50MB limits)
├── data/                 # Raw extracted images, engineered features, and training metrics (ignored)
├── notebooks/            # Jupyter notebooks for EDA and baseline model validation
├── src/
│   ├── data_pipeline/
│   │   ├── loader.py     # Image paths and target labeling utilities
│   │   └── preprocess.py # Feature extraction (Canny edge density, pixel intensities)
│   ├── models/
│   │   ├── baseline_model.py # Logistic Regression baseline training & metrics output
│   │   └── cnn_model.py      # Custom TensorFlow Keras CNN training & validation
│   └── api.py            # FastAPI endpoints for real-time baseline and CNN predictions
├── tests/
│   └── test_pipeline.py  # pytest suite validating preprocessing, training, and API endpoints
├── setup_data.py         # Reconstructs, unzips, and flattens the multi-part dataset archive
├── requirements.txt      # Python package dependencies
└── .gitignore            # Git exclusion rules for large datasets, model weights, and venv
```

---

## ⚙️ Quick Start & Setup

Follow these steps to set up the virtual environment, prepare the dataset, train models, and start the inference API.

### 1. Initialize Virtual Environment
Create a Python virtual environment and install all dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows Powershell)
venv\Scripts\Activate.ps1

# Activate virtual environment (Unix/macOS)
source venv/bin/activate

# Install required python packages
pip install -r requirements.txt
```

### 2. Reconstruct and Extract the Dataset
Since the Kaggle dataset is too large for GitHub's file size limits, it is committed as 8 split zip archives in `data_archive/`. Run the following script to automatically reconstruct, unzip, and clean the directory structure:
```bash
python setup_data.py
```
This merges the parts and creates a flat dataset directory structure under `data/seg_train` and `data/seg_test` directly.

---

## 🧠 Model Training

The pipeline provides two classification approaches: a lightweight baseline model and a deep learning CNN model.

### 1. Feature Engineering & Baseline Model
Extract hand-crafted visual features (RGB channel mean/standard deviations and Canny edge density) and train a Logistic Regression model:
```bash
# Extract features to data/train_features.csv
python src/data_pipeline/preprocess.py

# Train baseline model and save to src/models/baseline_model.joblib
python src/models/baseline_model.py
```

### 2. Deep Learning CNN Model
Train a custom 3-layer Convolutional Neural Network (CNN) in TensorFlow/Keras directly on the raw image pixels:
```bash
# Train CNN model and save weights to src/models/cnn_model.keras
python src/models/cnn_model.py
```
*Note: By default, the script trains for 5 epochs. This is optimized to complete quickly on CPU (approx. 2 minutes) while achieving ~77% accuracy.*

---

## 🚀 Model Serving (FastAPI API)

Expose the trained models as REST API endpoints for real-time predictions.

Start the FastAPI application using Uvicorn:
```bash
uvicorn src.api:app --reload
```

The API documentation will be available at:
* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Redoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### REST Endpoints
* **`GET /`**: Health check and status showing which models are loaded.
* **`POST /predict/baseline`**: Classifies an uploaded image using the baseline model.
* **`POST /predict/cnn`**: Classifies an uploaded image using the deep learning CNN model.

---

## 🧪 Automated Tests

Run the test suite using `pytest` to verify the components:
```bash
python -m pytest
```
This runs unit tests on feature extraction, model inference sanity checks, and API endpoints.