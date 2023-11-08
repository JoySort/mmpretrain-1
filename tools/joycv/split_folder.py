import os
import random
import shutil

src_root_path = "/nas/win_essd/seqee_training/selected/"
dst_root_path = "/nas/win_essd/seqee_training/selected_grouped/"
group_number = 300  # specify the number of groups for each category folder


# function to get all image files within a given directory
def get_image_files(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

# function to split images into groups and copy them to destination folders
def split_images(src_path, dst_path):
    # get all category folders
    category_folders = [folder for folder in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, folder))]

    # loop through category folders
    for category_folder in category_folders:
        # create the corresponding category folder in destination
        category_dst_folder = os.path.join(dst_path, category_folder)
        if not os.path.exists(category_dst_folder):
            os.makedirs(category_dst_folder)

        # get all image files in category folder
        image_files = get_image_files(os.path.join(src_path, category_folder))

        # determine the number of groups needed
        group_count = len(image_files) // group_number
        if len(image_files) % group_number != 0:
            group_count += 1

        # shuffle the image files
        #random.shuffle(image_files)

        # loop through groups
        for group_index in range(group_count):
            # create the corresponding group folder in destination
            group_dst_folder = os.path.join(category_dst_folder, f"group_{group_index+1}")
            if not os.path.exists(group_dst_folder):
                os.makedirs(group_dst_folder)

            # copy the image files to the group folder
            group_start = group_index * group_number
            group_end = (group_index + 1) * group_number
            for i, image_file in enumerate(image_files[group_start:group_end]):
                _, ext = os.path.splitext(image_file)
                original_name=os.path.basename(image_file)
                dst_file_name = f"{original_name}_{i+1}{ext}"
                dst_file_path = os.path.join(group_dst_folder, dst_file_name)
                shutil.copy(image_file, dst_file_path)

# split images in train folder
#split_images(os.path.join(src_root_path, 'correct'), dst_root_path)

# split images in test folder
split_images(os.path.join(src_root_path), dst_root_path)
