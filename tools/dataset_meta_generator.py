import os
import json
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split

def get_image_paths(directory, class_label, image_root):
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                relative_path = os.path.relpath(os.path.join(dirpath, filename), image_root)
                paths.append({'img_path': relative_path, 'gt_label': class_label})
    return paths

def split_data(paths, train_ratio=0.8):
    train_paths, test_paths = train_test_split(paths, train_size=train_ratio)
    return train_paths, test_paths

def create_metadata(image_root, train_ratio=0.8):
    train_dict = {'metainfo': {}, 'data_list': []}
    test_dict = {'metainfo': {}, 'data_list': []}

    first_level_dirs = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]

    train_dict['metainfo']['classes'] = test_dict['metainfo']['classes'] = tuple(first_level_dirs)

    log = {"train": {}, "test": {}}

    for i, dir_name in enumerate(first_level_dirs):
        full_path = os.path.join(image_root, dir_name)
        image_paths = get_image_paths(full_path, i, image_root)
        train_paths, test_paths = split_data(image_paths, train_ratio)

        train_dict['data_list'].extend(train_paths)
        test_dict['data_list'].extend(test_paths)

        log["train"][dir_name] = len(train_paths)
        log["test"][dir_name] = len(test_paths)

    log["train"]["total"] = len(train_dict['data_list'])
    log["test"]["total"] = len(test_dict['data_list'])

    train_dict["log"] = test_dict["log"] = log

    train_filename = os.path.join(image_root, f'train_{datetime.now().strftime("%Y%m%d%H%M%S")}.json')
    with open(train_filename, 'w') as f:
        json.dump(train_dict, f, indent=4)

    test_filename = os.path.join(image_root, f'test_{datetime.now().strftime("%Y%m%d%H%M%S")}.json')
    with open(test_filename, 'w') as f:
        json.dump(test_dict, f, indent=4)
    
    print(f"Train metadata saved to {train_filename}")
    print(f"Test metadata saved to {test_filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate metadata for image training.')
    parser.add_argument('image_root', help='Root directory for images.')
    parser.add_argument('--ratio', type=float, default=0.8, help='Train/Test ratio. Default is 0.8.')
    args = parser.parse_args()

    create_metadata(args.image_root, args.ratio)

if __name__ == '__main__':
    main()
