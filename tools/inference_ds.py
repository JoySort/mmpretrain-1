import os
import json
import time
import shutil
import mmcv
from mmcls.apis import init_model, inference_model

def load_coco_annotations(annotation_file):
    with open(annotation_file) as f:
        data = json.load(f)
    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    image_annotations = {img['file_name']: id_to_name[ann['category_id']] for img in data['images'] for ann in data['annotations'] if ann['image_id'] == img['id']}
    return image_annotations, id_to_name

def infer_and_compare(image_root, annotation_file, model_path, output_dir):
    # Load model
    model = init_model(model_path, device='cuda:0')

    # Load COCO annotations
    image_annotations, id_to_name = load_coco_annotations(annotation_file)

    for image_file in os.listdir(image_root):
        if image_file in image_annotations:
            img = mmcv.imread(os.path.join(image_root, image_file))
            result = inference_model(model, img)
            pred_label = result[0][0]
            pred_score = result[0][1]

            # Convert predicted label ID to category name
            pred_category = id_to_name[pred_label]

            true_category = image_annotations[image_file]

            if pred_category != true_category:
                # Save image in output directory
                new_filename = f"{os.path.splitext(image_file)[0]}_{pred_score:.3f}.png"
                new_directory = os.path.join(output_dir, pred_category)
                os.makedirs(new_directory, exist_ok=True)
                shutil.copy(os.path.join(image_root, image_file), os.path.join(new_directory, new_filename))

def main():
    parser = argparse.ArgumentParser(description='Infer and compare to annotations.')
    parser.add_argument('image_root', help='Root directory for images.')
    parser.add_argument('annotation_file', help='COCO annotation file.')
    parser.add_argument('model_path', help='Path to trained model.')
    parser.add_argument('output_dir', help='Directory to save images with differing predictions.')
    args = parser.parse_args()

    infer_and_compare(args.image_root, args.annotation_file, args.model_path, args.output_dir)

if __name__ == '__main__':
    main()
