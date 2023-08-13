from mmcls.apis import init_model, inference_model
import mmcv
import time,os
import json
import datetime
import numpy as np 

from clearml import Task
def validation_inference(validation_root_path,config_file,checkpoint_file,validation_discrepency_cp_path,correction_ops=False):
    import glob
    import time
    import platform
    from datetime import datetime as dt
    now = dt.now()
    timestr=now.strftime("%Y%m%d_%H:%M:%S")
    root_dir = validation_root_path

    image_list_map = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                image_path = os.path.join(dirpath, filename)
                relpath=os.path.relpath(image_path, root_dir)
                sub_dirname = relpath.split("/")[0]
                image_list_map[image_path] = sub_dirname

    
    validation_discrepency_cp_path=validation_discrepency_cp_path+"/"+"_".join(root_dir.split("/"))+timestr
    os.makedirs(validation_discrepency_cp_path, exist_ok=True)



    image_list=image_list_map.keys()
    unique_values = set(image_list_map.values())

    os.makedirs(validation_discrepency_cp_path, exist_ok=True)
    # for expected_labels in unique_values:
    #     os.makedirs(os.path.join(validation_discrepency_cp_path,expected_labels), exist_ok=True)

    print(f"expected labels {unique_values}")
    if len(image_list)==0:
        raise Exception(f"There are no image in {validation_root_path}/*.jpg")
    else:
        print(f"len(image_list) {len(image_list)}")

    model = init_model(config_file, checkpoint_file, device='cuda:0')
    model_name=os.path.basename(config_file).split(".")[0].split("_p")[0]

    validation_path_name=os.path.basename(validation_root_path)

    #task = Task.init(project_name='CLS_VALIDATION', task_name=f"{model_name}-{validation_path_name}-{platform.node()}-{timestr}")
    #Task.current_task().upload_artifact(name='config_file', artifact_object=config_file)
    
    
    validation_output_path_root=f"{validation_discrepency_cp_path}"
    counter=0
    bad_counter=0
    import shutil
    begin_time=time.time()
    inference_time_list=[]
    inference_counter={}
    expected_counter={}
    discrepancy_counter={}
    removed_file=[]
    for image_file in image_list:
        img = mmcv.imread(image_file)
        #img = mmcv.imrescale(img,scale=(500,1220,3))     
        #print(image_file)
        filename=os.path.basename(image_file)
        filename=filename.split(".")[0]
        foldername="_".join(image_file.split("/")[-4:-2])
        image_begin_time=time.time()
        inference_img=img
        result2,score = inference_model(model,inference_img )
        image_end_time=time.time()
        #print(f"single image timetook:{image_end_time-image_begin_time:.4f}")
        inference_time=image_end_time-image_begin_time
        inference_time_list.append(inference_time)
        inference_label=result2["pred_class"]
        if inference_label not in inference_counter:
            inference_counter[inference_label]=0
            #print(f"Found inference label {inference_label}")

        

        expected_label=image_list_map[image_file]
        if expected_label not in expected_counter:
            expected_counter[expected_label]=0
            
            
            #print(f"Found expected label {expected_label}")
        
        if expected_label not in discrepancy_counter:
            discrepancy_counter[expected_label]=0

        inference_counter[inference_label]=inference_counter[inference_label]+1
        expected_counter[expected_label]=expected_counter[expected_label]+1
        counter=counter+1
        

        

        if inference_label!=expected_label:
            discrepancy_counter[expected_label]=discrepancy_counter[expected_label]+1
            output_path=f"{validation_output_path_root}/wrong/{expected_label}"
            
        else:
            output_path=f"{validation_output_path_root}/correct/{expected_label}"

        os.makedirs(output_path,exist_ok=True)
        mmcv.imwrite(inference_img,f"{output_path}/{result2['pred_label']}-{result2['pred_class']}-{result2['pred_score']}{foldername}_{filename}.jpg")
        
        if correction_ops and (inference_label!=expected_label) :
            removed_file.append(image_file)
            os.remove(image_file)

        if counter % 1000 == 0:
            print(f"current counter {counter} discrepancy_counter {discrepancy_counter} \r\nexpected_counter {expected_counter}\r\n inference_counter:{inference_counter}")
    

    result_stats={}
    stats={}
    for item in discrepancy_counter:
        if item not in stats:
            stats[item]=0
        
        stats[item]=f"{round(discrepancy_counter[item]/expected_counter[item]*100,2)}%"



    time_took_avg=f"{(time.time()-begin_time)/counter:.4f}"
    inference_time_avg=np.mean(inference_time_list)
    result_stats["discrepancy_counter"]=discrepancy_counter
    result_stats["inference_counter"]=inference_counter
    result_stats["expected_counter"]=expected_counter
    result_stats["incorrect_rate"]=stats

    result_stats["total_counter"]=counter
    result_stats["model_name"]=model_name
    result_stats["time_took_avg"]=time_took_avg
    result_stats["inference_time_avg"]=inference_time_avg
    result_stats["total_counter"]=counter
    result_stats["config_file"]=config_file
    result_stats["checkpoint_file"]=checkpoint_file
    result_stats["validation_root_path"]=validation_root_path
    result_stats["removed_files"]=removed_file
    
    #Task.current_task().upload_artifact(name='result_stats', artifact_object=result_stats)
    json_object = json.dumps(result_stats, indent=4)
    print(f"{json_object} ")
    from datetime import datetime
    
    # get current date and time
    current_datetime = datetime.now()
    print("Current date & time : ", current_datetime)
    
    if correction_ops:
        print(f"Correction ops removed {len(removed_file)}")

    # convert datetime obj to string
    str_current_datetime = str(current_datetime)
    
    # create a file object along with extension
    file_name = str_current_datetime+".json"
    # Writing to sample.json
    with open(f"{validation_output_path_root}/{file_name}", "w") as outfile:
        outfile.write(json_object)


def chmod_recursive(path):
    import subprocess
    # build the command string with the path
    command = f"chmod -R og+rw {path}/../"

    # execute the command
    subprocess.run(command, shell=True, check=True)



config_file = '/opt/workspace/mmclassification/configs/joycv/resnet18_8xb16_package_cls_half_size_499_1220.py'
checkpoint_file = '/opt/workspace/mmclassification/work_dirs/resnet18_8xb16_package_cls_half_size_499_1220/best_accuracy_top-1_epoch_61.pth'

config_file = '/opt/workspace/mmclassification/work_dirs/repvgg-D2-package_with_validation_cls_half_size_499_1220/repvgg-D2-package_with_validation_cls_half_size_499_1220.py'
checkpoint_file = '/opt/workspace/mmclassification/work_dirs/repvgg-D2-package_with_validation_cls_half_size_499_1220/best_accuracy_top-1_epoch_13.pth'

config_file = '/opt/workspace/mmclassification/work_dirs/repvgg-A0-package_cls_half_size_499_1220/repvgg-A0-package_cls_half_size_499_1220.py'
checkpoint_file = '/opt/workspace/mmclassification/work_dirs/repvgg-A0-package_cls_half_size_499_1220/best_accuracy_top-1_epoch_17.pth'

config_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-02-26_03-16-28/out/chestnut_core_cls_config.py'
checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-02-26_03-16-28/out/best_accuracy_top-1_epoch_418.pth'


config_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-02-27_03-22-17/out/chestnut_core_repvgg.py'
checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-02-27_03-22-17/out/best_accuracy_top-1_epoch_116.pth'

config_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-07_16-55-20/out/chestnut_core_repvgg.py'
checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-07_16-55-20/out/best_accuracy_top-1_epoch_7.pth'

config_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_good_half_bd/2023-03-07_17-25-48/out/chestnut_core_good_half_bd.py'
checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_good_half_bd/2023-03-07_17-25-48/out/best_accuracy_top-1_epoch_86.pth'

config_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-10_19-20-46/out/chestnut_core_repvgg.py'
checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-10_19-20-46/out/best_accuracy_top-1_epoch_149.pth'

config_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-10_23-46-29/out/chestnut_core_repvgg.py'
checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-10_23-46-29/out/best_accuracy_top-1_epoch_361.pth'


config_file="/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-12_14-18-02/out/chestnut_core_repvgg.py"
checkpoint_file="/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-12_14-18-02/out/best_accuracy_top-1_epoch_125.pth"

config_file="/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-13_02-09-41/out/chestnut_core_repvgg.py"
checkpoint_file="/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-13_02-09-41/out/best_accuracy_top-1_epoch_204.pth"

config_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-15_12-02-39/out/chestnut_core_repvgg.py'
checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-15_12-02-39/out/best_accuracy_top-1_epoch_30.pth'

config_file = '/opt/workspace/mmpretrain-1/work_dirs/ximei_repvgg/config.py'
checkpoint_file = '/opt/workspace/mmpretrain-1/work_dirs/ximei_repvgg/best_accuracy_top1_epoch_165.pth'



validation_root = "/opt/workspace/imagedb/packs/trained/"
validation_root = "/opt/workspace/imagedb/packs/1202_untrained/"
validation_root = "/opt/workspace/imagedb/chestnut_core_sliced/formal_training_regrouped_0314/test"
validation_root = "/opt/workspace/imagedb/ximei/"

validation_discrepency_cp_path="/opt/workspace/imagedb/eval/"

validation_inference(validation_root,config_file,checkpoint_file,validation_discrepency_cp_path,correction_ops=False)
#chmod_recursive(validation_discrepency_cp_path)