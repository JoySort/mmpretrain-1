from mmcls.apis import init_model, inference_model
import mmcv
import time,os
# Specify the path to model config and checkpoint file
config_file = '/opt/workspace/mmcls_gitee/work_dirs/fresh_chestnut_repvgg/2023-02-16_23-42-27/out/config.py'
checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/fresh_chestnut_repvgg/2023-02-16_23-42-27/out/best_accuracy_top-1_epoch_88.pth'

#config_file = '/opt/workspace/mmcls_gitee/work_dirs/fresh_chestnut_repvgg/2023-02-13_08-32-28/out/config.py'
#checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/fresh_chestnut_repvgg/2023-02-13_08-32-28/out/best_accuracy_top-1_epoch_359.pth'
#config_file = '/opt/workspace/mmcls_gitee/work_dirs/fresh_chestnut_repvgg/2023-02-12_09-56-19_best_accuracy_95_top-1_epoch_177/out/config.py'
#checkpoint_file = '/opt/workspace/mmcls_gitee/work_dirs/fresh_chestnut_repvgg/2023-02-12_09-56-19_best_accuracy_95_top-1_epoch_177/out/best_accuracy_top-1_epoch_177.pth'


#checkpoint_file = 'c:/opt/workspace/mmcls/work_dirs/repvgg-A0-freshchestnut/2023_0125_2000/out/epoch_261.pth'

#config_file = '/opt/workspace/mmclassification/configs/joycv/repvgg-D2-package_with_validation_cls_half_size_499_1220.py'
#checkpoint_file = '/opt/workspace/mmclassification/work_dirs/repvgg-D2-package_with_validation_cls_half_size_499_1220/best_accuracy_top-1_epoch_13.pth'


# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:1')

#img_path="/opt/workspace/imagedb/package_cls_data/DatasetId_1725538_1669651240/Images/"
#ok_img_path="/opt/workspace/imagedb/input/fresh_chestnut_test_sample"
#bad_img_path="/opt/workspace/imagedb/package_cls_data/DatasetId_1733993_1670733467/sp2"

ok_img_paths=[]

ok_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/good/20230107break")
ok_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/good/20230107good")
ok_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/good/20230115good")
ok_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/good/20230115good2")
ok_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/good/20230115break")
#ok_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/good/20230115break2")
ok_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/good/20230115worm")

bad_img_paths=[]
#bad_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/break/simple")
# bad_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/break/simple2")
# bad_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/break/simple3")
bad_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/worm/first")

ok_img_paths_test=[]

ok_img_paths_test.append("C:/opt/workspace/imagedb/freshchestnut/test/good/20230107break")
ok_img_paths_test.append("C:/opt/workspace/imagedb/freshchestnut/test/good/20230107good")
ok_img_paths_test.append("C:/opt/workspace/imagedb/freshchestnut/test/good/20230115good")
ok_img_paths_test.append("C:/opt/workspace/imagedb/freshchestnut/test/good/20230115break")
# #ok_img_paths.append("C:/opt/workspace/imagedb/freshchestnut/train/good/20230115break2")
#ok_img_paths_test.append("C:/opt/workspace/imagedb/freshchestnut/test/good/20230115worm")

bad_img_paths_test=[]
#bad_img_paths_test.append("C:/opt/workspace/imagedb/freshchestnut/test/break/simple")
#bad_img_paths_test.append("C:/opt/workspace/imagedb/freshchestnut/test/break/simple3")
bad_img_paths_test.append("C:/opt/workspace/imagedb/freshchestnut/test/worm/first")

import shutil
import datetime
def inference(image_path,targetID,recursive=False):
    import glob
    import time
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

    output_path=f"/opt/workspace/imagedb/freshchestnut/inference_result/{checkpoint_name}_{image_path_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_path,exist_ok=True)
    counter=0
    stats_percent={}
    stats_counter={}
    import shutil
    begin_time=time.time()
    total=len(image_list)
    for image_file in image_list:
        img = mmcv.imread(image_file)
        #print(image_file)
        #img = mmcv.imrescale(img,scale=(500,1220,3))     
        #print(image_file)
        filename=os.path.basename(image_file)
        filename="_".join(filename.split(".")[:-1])
        image_begin_time=time.time()
        inference_img=img
        result2 = inference_model(model,inference_img )
        image_end_time=time.time()
        #print(f"single image timetook:{image_end_time-image_begin_time:.4f}")
        counter=counter+1
        
        class_label=result2['pred_class']
        if (class_label not in stats_counter):
            stats_counter[class_label]=0
            stats_percent[class_label]=0

        stats_counter[class_label]=stats_counter[class_label]+1 
        stats_percent[class_label]=f"{stats_counter[class_label]/counter*100:.2f}%"

        mmcv.imwrite(inference_img,f"{output_path}/{result2['pred_class']}/{filename}.jpg")

        if(counter % 10 ==0 or counter == total):
            print(f"total image: {counter}, stats: [{stats_counter}]percent:{stats_percent} avg timetook:{(time.time()-begin_time)/counter:.4f}")
        
    

# for ok_img_path in ok_img_paths:
#     inference(ok_img_path,1)

# for bad_img_path in bad_img_paths:
#     inference(bad_img_path,0)

#for ok_img_path_test in ok_img_paths_test:
#    inference(ok_img_path_test,1)
#for bad_img_path_test in bad_img_paths_test:
#    inference(bad_img_path_test,0)

#inference("c:/opt/workspace/imagedb/freshchestnut/origin/20230115/break20230115",1)
inference("/opt/workspace/imagedb/slice/debug_draw_sliceFeb-16-2023_23181676560717/sliced_result/",1,recursive=True)#good
#inference("/opt/workspace/imagedb/slice/debug_draw_sliceFeb-16-2023_23281676561323/sliced_result/",1,recursive=True)#bad
#inference("/opt/workspace/imagedb/freshchestnut/test/worm/20230216_fixed_shot_bazhou_pure_wormhole_for_test/",1,recursive=True)#0-good,1-bad


