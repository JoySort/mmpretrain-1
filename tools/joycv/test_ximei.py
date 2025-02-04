import mmcv
import time,os
import json
import numpy as np
import shutil
import datetime
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.fileio import dump
from rich import print_json

from mmpretrain.apis import ImageClassificationInferencer

def inference(image_path,inferencer,targetID,output_path,recursive=False,split=False):
    import glob
    import time
    if(not os.path.exists(image_path)):
        raise Exception("image_path does not exist!")
    png_pattern=f"{image_path}/*.png"
    bmp_pattern=f"{image_path}/*.bmp"
    jpg_pattern=f"{image_path}/*.jpg"
    if(recursive):
        png_pattern=f"{image_path}/**/*.png"
        bmp_pattern=f"{image_path}/**/*.bmp"
        jpg_pattern=f"{image_path}/**/*.jpg"

    image_list=glob.glob(png_pattern ,recursive=recursive)
    image_list.extend(glob.glob(bmp_pattern,recursive=recursive))
    image_list.extend(glob.glob(jpg_pattern,recursive=recursive))


    image_path_name="_".join(image_path.split("/"))

    #print(f"inference list {image_list}")
    checkpoint_name=os.path.basename(checkpoint_file).split(".")[0]

    
    os.makedirs(output_path,exist_ok=True)

    sub_folders=create_folders(0,1,0.1,output_path,split)
    
    counter=0
    stats_percent={}
    stats_counter={}
    import shutil
    begin_time=time.time()
    inference_time=[];
    total=len(image_list)
    for image_file in image_list:
        img = mmcv.imread(image_file)
        #print(image_file)
        #img = mmcv.imrescale(img,scale=(500,1220,3))     
        #print(image_file)
        filename=os.path.basename(image_file)
        dirname=os.path.dirname(image_file)
        dirname=""#"_".join(dirname.split("/"))[:-1]
        filename="_".join(filename.split(".")[:-1])
        image_begin_time=time.time()
        inference_img=img
        result = inferencer(inference_img, show=False)[0]
        print(result)
        # show the results
        scores = result.pop('pred_scores')  # pred_scores is too verbose for a demo.
        #score=result
        
        print(scores)
        sorted_indices = np.argsort(scores)[::-1]

        # the index of the second highest score is the second element in this sorted list of indices
        top2_score_index = sorted_indices[1]

        # and the second highest score is
        top2_score = scores[top2_score_index]

        image_end_time=time.time()
        inference_time.append(image_end_time-image_begin_time)
        #print(f"single image timetook:{image_end_time-image_begin_time:.4f}")
        counter=counter+1
        #model.CLASSES[result['pred_label']]
        
        class_label=result['pred_class']
        class_score=result['pred_score']
        second_label=classes[top2_score_index]
        second_score=top2_score
        

        if (class_label not in stats_counter):
            stats_counter[class_label]=0
            stats_percent[class_label]=0


        stats_counter[class_label]=stats_counter[class_label]+1 
        stats_percent[class_label]=f"{stats_counter[class_label]/counter*100:.2f}%"
        
        sub_folder_index=int(round(class_score,1)*10)
        file_path=sub_folders[sub_folder_index]
        os.makedirs(f"{file_path}/{result['pred_class']}/{second_label}/", exist_ok=True)
        mmcv.imwrite(inference_img,f"{file_path}/{result['pred_class']}/{second_label}/{class_label}-{class_score:.5f}-{second_label}{second_score:.5f}-{filename}.png")

        if(counter % 10 ==0 or counter == total):
            print(f"total image: {counter}, stats: [{stats_counter}]percent:{stats_percent} avg timetook:{(time.time()-begin_time)/counter:.4f}")
            # Calculate max, min, and mean
            max_time = max(inference_time)
            min_time = min(inference_time)
            mean_time = np.mean(inference_time)  # You can also use sum(inference_time) / len(inference_time)

            # Print the statistics
            print("Stats:")
            print("Count: ", len(inference_time))
            print("Max: ", max_time)
            print("Min: ", min_time)
            print("Mean: ", mean_time)


    # Calculate max, min, and mean
    max_time = max(inference_time)
    min_time = min(inference_time)
    mean_time = np.mean(inference_time)  # You can also use sum(inference_time) / len(inference_time)

    # Print the statistics
    print("Final Stats:")
    print("Count: ", len(inference_time))
    print("Max: ", max_time)
    print("Min: ", min_time)
    print("Mean: ", mean_time)
    stats_obj=dict(
        count=stats_counter,
        percentile=stats_percent
    )
    json_result=json.dumps(stats_obj, indent=4) 
    print(f"{json_result}") 
    with open(f"{output_path}/stats.json", "w") as outfile:
        outfile.write(
                json_result
        )
    print(f"output_path is {output_path}")
    file_count_obj = json.dumps(count_jpg_files(output_path), indent=4)  
    print(file_count_obj)

def chmod_recursive(path):
    import subprocess
    # build the command string with the path
    command = f"chmod -R og+rw {path}/../"
    # execute the command
    subprocess.run(command, shell=True, check=True)
def create_folders(start_num, end_num, step, root_path,split):
    # Create an empty list to store the numbers
    nums = []
    
    # Initialize the loop variable to the start number
    num = start_num
    
    # Loop while the current number is less than or equal to the end number
    while num <= end_num:
        # Append the current number rounded to one decimal place to the list
       
        
        # Create a folder with the name of the current number appended to the root path
        if(split):
            folder_path = os.path.join(root_path, str(round(num, 1)))
        else:
            folder_path = root_path
        print(f"creating folder {folder_path}")
        nums.append(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        
        # Increment the current number by the step size
        num += step
    
    # Return the list of numbers
    return nums   
def count_jpg_files(root_path):
    import glob
    import json
    # Create an empty dictionary to store the file counts
    file_counts = {}
    total=0
    # Iterate over the directory tree rooted at the given path
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Use glob to find all .jpg files in the current directory
        jpg_files = glob.glob(os.path.join(dirpath, '*.png'))
        dic_index=dirpath.replace(root_path, "")
        # Add the count of .jpg files to the dictionary for the current directory
        file_counts[dic_index] = len(jpg_files)
        total=total+ len(jpg_files)

    file_counts["total"]=total

    stats={}
    for item in file_counts:
        key=item.split("/")[0]
        if(key not in stats ):
            stats[key]=0
        stats[key]=stats[key]+file_counts[item]
    sorted_keys = sorted(stats.keys())
    sorted_dict = {key:stats[key] for key in sorted_keys}
    json_object = json.dumps(sorted_dict, indent=4)    
    print(f"file_counts:{json_object}")
    # Return the dictionary of file counts
    

    return file_counts
# for ok_img_path in ok_img_paths:
#     inference(ok_img_path,1)

# for bad_img_path in bad_img_paths:
#     inference(bad_img_path,0)

#for ok_img_path_test in ok_img_paths_test:
#    inference(ok_img_path_test,1)
#for bad_img_path_test in bad_img_paths_test:
#    inference(bad_img_path_test,0)

src_path="/opt/workspace/imagedb/ximei/"
src_path="/nas/win_essd/西梅/good150_slice/1691709712/slice/"
src_path="/nas/win_essd/UAE_sliced_256/mid_east_dates/sau"

class_config_file="/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-20_09-49-39/out/class_config.json"
output_path=f"/nas/win_essd/西梅/inference/"
output_path=f"/nas/win_essd/pd_inference/"

config_file = '/opt/workspace/mmpretrain-1/work_dirs/palmdate_cls_MobileViT-image_size_256_batch_size32-datasetintermediate_model_candidate_correction-batchsize_32-maxep_1000-joysort-ai-server/2023-08-21_00-53-41/out/20230821_005406/vis_data/config.py'
checkpoint_file = '/opt/workspace/mmpretrain-1/work_dirs/palmdate_cls_MobileViT-image_size_256_batch_size32-datasetintermediate_model_candidate_correction-batchsize_32-maxep_1000-joysort-ai-server/2023-08-21_00-53-41/out/epoch_300.pth'

try:
        pretrained = checkpoint_file or True
        inferencer = ImageClassificationInferencer(
            config_file, pretrained=pretrained)
except ValueError:
    raise ValueError(
        f'Unavailable model "{checkpoint_file}", you can specify find a model '
        'name or a config file or find a model name from '
        'https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html#all-checkpoints'  # noqa: E501
    )
classes=inferencer.classes
print("classes",classes)   
image_path_name="_".join(src_path.split("/"))
checkpoint_name=os.path.basename(checkpoint_file).split(".")[0]
actual_output_path=f"{output_path}{checkpoint_name}_{image_path_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
inference(src_path,inferencer,1,actual_output_path,recursive=True,split=False)#0-good,1-bad
#chmod_recursive(actual_output_path)

#file_count=count_jpg_files("/opt/workspace/imagedb/chestnut_core/inference_result/best_accuracy_top-1_epoch_204__opt_workspace_imagedb_slice_sliced_result_raw_0308/2023-03-13_10-22-35/")
json_object = json.dumps(count_jpg_files(actual_output_path), indent=4)  
print(json_object)