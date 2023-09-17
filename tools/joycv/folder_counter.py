import os

def count_images_in_folder(folder_path):
    count = 0
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

    # Count images in the current folder
    for file_ in os.listdir(folder_path):
        if file_.lower().endswith(('.jpg', '.png', '.bmp')):
            count += 1

    # Count images in subfolders and add to the current folder's count
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        subfolder_count = count_images_in_folder(subfolder_path)
        count += subfolder_count

    # Print the hierarchy and count
    relative_path = os.path.relpath(folder_path, start_path)
    indent_level = relative_path.count(os.path.sep)
    print('  ' * indent_level + f"{folder_path} - {count} images")
    
    return count

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <source path>")
        sys.exit(1)

    start_path = sys.argv[1]
    print(f"Image count in folder hierarchy starting from {start_path}:")
    count_images_in_folder(start_path)
