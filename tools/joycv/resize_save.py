from PIL import Image
import os

def resize_and_save(path, target_size=(224, 224)):
    # Get all files in path and subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if file is an image
            if file.endswith(('png', 'jpg', 'jpeg')):
                # Open image and get its size
                
                img_path = os.path.join(root, file)
                
                img = Image.open(img_path)
                size = img.size

                # Check if size matches target size
                if size != target_size:
                    print(f"processing {img_path} and  size not fit {size} converting")
                    # Resize image
                    img = img.resize(target_size)

                    # Overwrite original file
                    img.save(img_path)


resize_and_save("/opt/workspace/imagedb/chestnut_core_sliced/train_to_select/")