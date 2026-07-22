import os
import cv2
import pandas as pd
import numpy as np

# --- 1. Define Constants and Paths ---
DATA_DIR = 'data/seg_train'
OUTPUT_CSV = 'data/train_features.csv'
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- 2. Feature Extraction Function ---
def extract_features_from_image(image_path):
    # Load image (OpenCV loads images in BGR format by default)
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Resize for consistent processing (Intel images are 150x150, but good practice to resize)
    image = cv2.resize(image, (64, 64)) 

    # 1. Feature: Mean Pixel Intensity (Advanced Preprocessing)
    # The image is split into channels.
    b, g, r = cv2.split(image)
    
    # 2. Feature: Edge Density (Advanced Feature Engineering)
    # Convert to grayscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Canny edge detector
    edges = cv2.Canny(gray_image, 100, 200)
    # Edge density is the ratio of white pixels (edges) to total pixels
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # Return a flat list of descriptive features
    features = [
        np.mean(r), np.mean(g), np.mean(b),
        np.std(r), np.std(g), np.std(b),
        edge_density
    ]
    return features

# --- 3. Main Processing Logic ---
def generate_features_csv():
    all_data = []
    
    # Process images for each class
    for label_id, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        print(f"Processing images in: {class_name}")

        for filename in os.listdir(class_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_dir, filename)
                features = extract_features_from_image(image_path)
                
                if features:
                    # Append features and the target label (numeric and string)
                    features.extend([class_name, label_id])
                    all_data.append(features)

    # Create DataFrame with descriptive column names
    column_names = [
        'mean_R', 'mean_G', 'mean_B', 
        'std_R', 'std_G', 'std_B', 
        'edge_density', 
        'label_name', 'label_id'
    ]
    df = pd.DataFrame(all_data, columns=column_names)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFeature engineering complete. Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    # Ensure you have installed opencv-python: pip install opencv-python
    generate_features_csv()