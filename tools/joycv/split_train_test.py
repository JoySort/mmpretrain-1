import os
import random
import shutil

def copy_images(src, dest_root, percentage=20.0, dry_run=True):
    total_count = 0
    subfolders = [f.name for f in os.scandir(src) if f.is_dir()]

    # Get image files in the current folder
    image_files = [file_ for file_ in os.listdir(src) if file_.lower().endswith(('.jpg', '.png', '.bmp'))]
    count = len(image_files)

    # Calculate the number of images to copy for training and testing
    num_test = int(count * (percentage / 100.0))
    num_train = count - num_test

    # Randomly select and copy images
    random.shuffle(image_files)
    for i, file_ in enumerate(image_files):
        src_file_path = os.path.join(src, file_)
        relative_path = os.path.relpath(src, start_path)
        
        if i < num_test:
            dest_folder = os.path.join(dest_root, "test", relative_path)
        else:
            dest_folder = os.path.join(dest_root, "train", relative_path)

        dest_file_path = os.path.join(dest_folder, file_)

        print(f"copying {src_file_path} to {dest_file_path}")  # Debugging line
        if not dry_run:
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(src_file_path, dest_file_path)
    
    # Traverse subfolders
    for subfolder in subfolders:
        src_folder_path = os.path.join(src, subfolder)
        subfolder_count = copy_images(src_folder_path, dest_root, percentage, dry_run)
        count += subfolder_count

    return count

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Copy images for ML training and testing.')
    parser.add_argument('source_path', type=str, help='Source path')
    parser.add_argument('destination_path', type=str, help='Destination path')
    parser.add_argument('--percentage', type=float, default=20.0, help='Percentage of images to use for testing')
    parser.add_argument('--dry-run', type=bool, default=False, help='Dry run mode')

    args = parser.parse_args()
    start_path = args.source_path

    print("Running in Dry Run mode: ", args.dry_run)
    input("Press any key to continue or Ctrl+C to interrupt...")

    copy_images(args.source_path, args.destination_path, args.percentage, args.dry_run)
