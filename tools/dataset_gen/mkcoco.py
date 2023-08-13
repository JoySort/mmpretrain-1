from struct_def import *
import glob

folder_list=[]

folder_list.append(folder_classname_map('f:\\sample\\pack_big\\pack_train\\V1\\good','good'))
folder_list.append(folder_classname_map('f:\\sample\\pack_big\\pack_train\\V1\\empty','empty'))
folder_list.append(folder_classname_map('f:\\sample\\pack_big\\pack_train\\V1\\bad','bad'))
folder_list.append(folder_classname_map('f:\\sample\\pack_big\\pack_train\\V2\\good','good'))

folder_list.append(folder_classname_map('d:\\pics\\V3\\good','good'))
folder_list.append(folder_classname_map('d:\\pics\\V4\\good','good'))
folder_list.append(folder_classname_map('d:\\pics\\V3\\empty','empty'))
folder_list.append(folder_classname_map('d:\\pics\\V3\\bad','bad'))


folder_list.append(folder_classname_map('f:\\sample\\pack_big\\pack_val\\V1\\good','good'))
folder_list.append(folder_classname_map('f:\\sample\\pack_big\\pack_val\\V1\\empty','empty'))
folder_list.append(folder_classname_map('f:\\sample\\pack_big\\pack_val\\V1\\bad','bad'))
folder_list.append(folder_classname_map('f:\\sample\\pack_big\\pack_val\\V2\\good','good'))

class_img_dict={}

for img_folder in folder_list:
    folder=img_folder.folder
    class_name=img_folder.classname
    if class_name not in class_img_dict:
        class_img_dict[class_name]=[]
    img_file_list=glob.glob(f"{folder}/*.bmp")
    for img_file in img_file_list:
         class_img_dict[class_name].append(img_file)

import random
train_img_dict={}
val_img_dict={}

for key in class_img_dict.keys():
    random.shuffle(class_img_dict[key])
    img_count=len(class_img_dict[key])
    val_img_count=img_count//7
    train_img_count=img_count-val_img_count
    train_img_dict[key]=class_img_dict[key][0:train_img_count]
    val_img_dict[key]=class_img_dict[key][-val_img_count:]

train_img_class_list=[]
val_img_class_list=[]


category_list=[]

index=0
for key in train_img_dict.keys():
    for img_file in train_img_dict[key]:
        train_img_class_list.append(img_classindex_map(img_file,index))
    cat_item={}
    cat_item['id']=index
    cat_item['name']=key
    category_list.append(cat_item)
    index=index+1
random.shuffle(train_img_class_list)

index=0
for key in val_img_dict.keys():
    for img_file in val_img_dict[key]:
        val_img_class_list.append(img_classindex_map(img_file,index))
    index=index+1
random.shuffle(val_img_class_list)

coco_train={}
coco_val={}



output_path="./output"


import os
if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(os.path.join(output_path,"train")):
    os.mkdir(os.path.join(output_path,"train"))

if not os.path.exists(os.path.join(output_path,"train/Annotations")):
    os.mkdir(os.path.join(output_path,"train/Annotations"))

if not os.path.exists(os.path.join(output_path,"train/Images")):
    os.mkdir(os.path.join(output_path,"train/Images"))

if not os.path.exists(os.path.join(output_path,"val")):
    os.mkdir(os.path.join(output_path,"val"))

if not os.path.exists(os.path.join(output_path,"val/Annotations")):
    os.mkdir(os.path.join(output_path,"val/Annotations"))

if not os.path.exists(os.path.join(output_path,"val/Images")):
    os.mkdir(os.path.join(output_path,"val/Images"))
import json
import shutil
coco_train["images"]=[]
coco_train["annotations"]=[]
img_id=0

print(len(train_img_class_list))
for img_class in train_img_class_list:
    filename=os.path.basename(img_class.img)
    class_id=img_class.classindex
    shutil.copy(img_class.img,os.path.join('./output/train/Images',filename))

    #{"file_name": "20221210_181428_131_001574.bmp", "height": 500, "width": 1220, "id": 1}
    
    img_item={}
    img_item["file_name"]=filename
    img_item["height"]=500
    img_item["width"]=1220
    img_id=img_id+1
    img_item["id"]=img_id
    coco_train["images"].append(img_item)
    #{"segmentation": null, "area": 1.1, "iscrowd": 0, "bbox": null, "category_id": 0, "image_id": 1436, "id": 1436}, 
    anno_item={}
    anno_item["segmentation"]=None
    anno_item["area"]=1.1
    anno_item["iscrowd"]=0
    anno_item["bbox"]=None
    anno_item["category_id"]=class_id
    anno_item["image_id"]=img_id
    anno_item["id"]=img_id
    coco_train["annotations"].append(anno_item)




coco_val["images"]=[]
coco_val["annotations"]=[]

print(len(val_img_class_list))
img_id=0
for img_class in val_img_class_list:
    filename=os.path.basename(img_class.img)
    class_id=img_class.classindex
    shutil.copy(img_class.img,os.path.join('./output/val/Images',filename))

    #{"file_name": "20221210_181428_131_001574.bmp", "height": 500, "width": 1220, "id": 1}
    
    img_item={}
    img_item["file_name"]=filename
    img_item["height"]=500
    img_item["width"]=1220
    img_id=img_id+1
    img_item["id"]=img_id
    coco_val["images"].append(img_item)
    #{"segmentation": null, "area": 1.1, "iscrowd": 0, "bbox": null, "category_id": 0, "image_id": 1436, "id": 1436}, 
    anno_item={}
    anno_item["segmentation"]=None
    anno_item["area"]=1.1
    anno_item["iscrowd"]=0
    anno_item["bbox"]=None
    anno_item["category_id"]=class_id
    anno_item["image_id"]=img_id
    anno_item["id"]=img_id
    coco_val["annotations"].append(anno_item)



coco_train["info"]='info'
coco_val["info"]='info'

coco_train["licenses"]="licenses"
coco_val["licenses"]="licenses"

coco_train["image_nums"]=len(train_img_class_list)
coco_val["image_nums"]=len(val_img_class_list)


coco_train["categories"]=category_list
coco_val["categories"]=category_list

file = open(os.path.join(output_path,'train/Annotations/coco_info.json'),'w')
json.dump(coco_train,file)
    
file = open(os.path.join(output_path,'val/Annotations/coco_info.json'),'w')
json.dump(coco_val,file)

#python ./tools/train.py ./configs/joycv/repvgg-A0-package_cls_half_size_500_1220.p
