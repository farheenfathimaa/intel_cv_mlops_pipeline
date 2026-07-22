import os
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. Define Paths and Constants ---
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'seg_train')
TEST_DIR = os.path.join(DATA_DIR, 'seg_test')
MODEL_PATH = 'src/models/cnn_model.keras'
METRICS_PATH = 'data/cnn_metrics.txt'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 5  # Small number of epochs to train quickly on CPU, but enough to show learning

def train_cnn_model():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print("Error: Dataset directories not found. Please run setup_data.py first.")
        return

    print("Loading datasets...")
    # Load training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Load validation dataset from training directory
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Optimize datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Define simple CNN architecture
    # Normalize pixel values to [0, 1] using Rescaling layer
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(6)  # 6 classes
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    print("Starting CNN training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    print("Training complete.")

    # Evaluate on test set
    print("Evaluating CNN model on test set...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"\nCNN Test Accuracy: {accuracy:.4f}")

    # Persist the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"CNN model persisted to: {MODEL_PATH}")

    # Save metrics
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        f.write(f"Test Accuracy: {accuracy}\n")
        f.write(f"Test Loss: {loss}\n")
    print(f"CNN metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    train_cnn_model()
