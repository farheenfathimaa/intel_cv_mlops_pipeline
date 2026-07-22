import os

# Define the 6 class names based on the Intel dataset
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def get_image_paths(base_dir='data'):
    """
    Collects image paths and corresponding labels from the structured dataset directories.
    Assumes the structure: base_dir/seg_train/class_name/image.jpg
    """
    train_dir = os.path.join(base_dir, 'seg_train')
    
    if not os.path.isdir(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return []

    image_data = []
    
    for class_name in CLASSES:
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            label_id = CLASSES.index(class_name)
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_path, filename)
                    image_data.append({
                        'path': image_path,
                        'label_name': class_name,
                        'label_id': label_id
                    })
    
    print(f"Found {len(image_data)} images for training data setup.")
    return image_data

if __name__ == "__main__":
    # Example usage (will look for the 'data/seg_train' directory)
    _ = get_image_paths()
