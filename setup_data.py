import os
import zipfile
import shutil

def combine_and_extract():
    archive_dir = 'data_archive'
    output_zip = 'data.zip'
    data_dir = 'data'
    
    # 1. Concatenate multipart zip if archive exists
    if os.path.exists(archive_dir):
        parts = sorted([f for f in os.listdir(archive_dir) if f.startswith('data.zip.')])
        if parts:
            print(f"Combining {len(parts)} parts of the dataset from '{archive_dir}'...")
            with open(output_zip, 'wb') as outfile:
                for part in parts:
                    part_path = os.path.join(archive_dir, part)
                    with open(part_path, 'rb') as infile:
                        outfile.write(infile.read())
            
            print("Extracting combined zip to 'data' directory...")
            os.makedirs(data_dir, exist_ok=True)
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            os.remove(output_zip)
            print("Extraction finished.")
        else:
            print("No split zip files found in data_archive. Assuming data is already unzipped.")
    else:
        print("data_archive folder not found. Assuming data is already unzipped.")

    # 2. Flatten nested directories if they exist
    flatten_nested_dirs(data_dir)

def flatten_nested_dirs(data_dir):
    print("Checking for nested folders to flatten...")
    for sub in ['seg_train', 'seg_test']:
        sub_path = os.path.join(data_dir, sub)
        if os.path.exists(sub_path):
            nested_path = os.path.join(sub_path, sub)
            if os.path.exists(nested_path) and os.path.isdir(nested_path):
                print(f"Flattening nested directory: {nested_path} -> {sub_path}")
                # Move all contents from nested_path to sub_path
                for item in os.listdir(nested_path):
                    src_item = os.path.join(nested_path, item)
                    dst_item = os.path.join(sub_path, item)
                    # Handle existing items if any
                    if os.path.exists(dst_item):
                        if os.path.isdir(dst_item):
                            shutil.rmtree(dst_item)
                        else:
                            os.remove(dst_item)
                    shutil.move(src_item, dst_item)
                # Remove the now empty nested directory
                os.rmdir(nested_path)
    print("Dataset directory structure is clean and flat.")

if __name__ == "__main__":
    combine_and_extract()
