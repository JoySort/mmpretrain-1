import os
import json
from colorama import Fore, Style
from termcolor import colored

def generate_coco_annotation_from_folder(image_path_with_class_label, annotation_dest_path):
    """
    This function takes the input folder path and generates a COCO annotation file 
    based on the images and class labels present in it. The output JSON file is saved
    in the output directory path provided.
    
    Parameters:
    image_path_with_class_label (str): path of the folder containing images with class labels
    annotation_dest_path (str): path of the output directory where the annotation file is saved
    """

    #normalize the image path
    if(not image_path_with_class_label.endswith("/")):
        image_path_with_class_label=image_path_with_class_label+"/"
    print(f"======={Fore.LIGHTGREEN_EX}COCO annotation generation{Style.RESET_ALL}======")
    print(f"Start processing folder{Fore.LIGHTBLUE_EX}{image_path_with_class_label}{Style.RESET_ALL} for coco annoatation")

    # Step 2: Read only the top folders to get category names
    categories = []
    top_level_folders = sorted([name for name in os.listdir(image_path_with_class_label) if os.path.isdir(os.path.join(image_path_with_class_label, name))])
    for folder in top_level_folders:
        category = {"name": folder,"id":top_level_folders.index(folder) }
        categories.append(category)
        
    # Step 4: Walk through the folders and their sub-folders to get image paths
    images = []
    annotations = []
    class_counts = {}
    folder_statistic={}
    for root, dirs, files in os.walk(image_path_with_class_label):
        # Include only image files with extensions bmp, jpg, png, tiff
        file_types = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
        image_files = [f for f in files if f.lower().endswith(file_types)]
        
        # If there are image files in the folder, get their paths
        if image_files:
            # Get the category name from the root folder of the image
            
            category_names = [cn for cn in categories if cn['name'] in root.split("/")]
            # Check if the category name exists in the top level folders
            if (len(category_names)<=0):
                continue
            category_name = category_names[-1]['name']
            # Count the images per class
            if category_name not in class_counts:
                class_counts[category_name] = 0
            class_counts[category_name] += len(image_files)
            relative_root_split=root.split(image_path_with_class_label)
            relative_root="".join(relative_root_split)
            if (relative_root not in folder_statistic):
                folder_statistic[relative_root]=0
            folder_statistic[relative_root]+=len(image_files)
            # Get the image paths and add them to the images list
            for image_file in image_files:
                image_path = os.path.join(relative_root, image_file)
                image_info = {
                    "file_name": image_path,
                    "height": 224, # Set the height and width to 0 for now
                    "width": 224,
                    "id": len(images)+1, # Use image index as ID
                    
                }
                
                annotation_info={
                        
                    "segmentation": None,
                    "area": 1.1,
                    "iscrowd": 0,
                    "bbox": None,
                    "category_id": top_level_folders.index(category_name), # Use index of category in categories list as ID
                    "image_id": len(images)+1,
                    "id": len(images)+1
                    
                }
                images.append(image_info)
                annotations.append(annotation_info)
                
    # Step 5: Generate the COCO annotation using python json lib
    annotation_data = {
        "info":"info",
        "images": images,
        "licenses":"licenses",
        "image_nums": len(images),
        "categories": categories,
        "annotations": annotations
    }
    #print(categories)

    anno_folder=os.path.dirname(annotation_dest_path)
    print(f"Examin anno path:{annotation_dest_path}. containing folder{anno_folder}")
    os.makedirs(anno_folder,exist_ok=True)

    with open(annotation_dest_path, 'w') as f:
        json.dump(annotation_data, f)
        
    
    # Step 6: Store class counts in memory
    #class_counts_in_memory = {category: class_counts[category] for category in top_level_folders}

    
    print(colored(json.dumps(folder_statistic, indent=4), 'cyan'))
    print(f"Total image count:{Fore.GREEN}{len(images)}{Style.RESET_ALL}")
    print(colored(json.dumps(class_counts, indent=4), 'cyan'))
    return categories,folder_statistic,class_counts



if (__name__ == "__main__") :
    cat=generate_coco_annotation_from_folder(
        "/opt/images/train_cls_gd_inference/correct/",
        "/opt/images/train_cls_gd_inference/correct/anno.json"
    )
    print(cat)

