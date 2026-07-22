import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# --- 1. Define Paths and Constants ---
FEATURES_CSV = 'data/train_features.csv'
MODEL_PATH = 'src/models/baseline_model.joblib'
TARGET_COLUMN = 'label_id'

# --- 2. Training Function ---
def train_baseline_model():
    # Load data from Phase 2
    df = pd.read_csv(FEATURES_CSV)
    
    # Define Features (X) and Target (y)
    X = df[['mean_R', 'mean_G', 'mean_B', 'std_R', 'std_G', 'std_B', 'edge_density']].values
    y = df[TARGET_COLUMN].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    # Create a Scikit-learn Pipeline (preprocessing + model)
    # The pipeline ensures that the scaling applied to training data is also applied to production data
    pipeline = Pipeline([
        ('scaler', StandardScaler()),                 # Preprocessing: Scaling features
        ('model', LogisticRegression(max_iter=500))  # Model: Simple Logistic Regression
    ])

    # Train the model
    print("Starting baseline model training...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\nBaseline Model Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score (Macro): {f1:.4f}")
    
    # Persist the model (key for MLOps)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Baseline model persisted to: {MODEL_PATH}")
    
    # Save metrics for later MLOps tracking/comparison (Phase 5)
    with open('data/baseline_metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")

if __name__ == "__main__":
    train_baseline_model()