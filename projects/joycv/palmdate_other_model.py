#_base_ = 'mmpretrain::repvgg/repvgg-A0_8xb32_in1k.py' #1.36 - 72.37
#_base_ = 'mmpretrain::levit/levit-256_8xb256_in1k.py' #1.14 - 81.59
_base_ = 'mmpretrain::mobilevit/mobilevit-small_8xb128_in1k.py'





train_max_epochs=1000

batch_size=32 
image_size=256
data_root="/nas/win_essd/UAE_sliced_256/pd_train_candidate/intermediate_model_candidate_correction/"
#data_root="/nas/win_essd/西梅/训练样本/"

checkpoint_config = dict(interval=25)
train_cfg = dict(by_epoch=True, max_epochs=train_max_epochs, val_interval=1)

def extract_category_info(data_root):
    import os

    subfolder_names = []
    
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        
        if os.path.isdir(item_path):  # Check if it's a directory
            subfolder_names.append(item)
    subfolder_names.sort()
    return subfolder_names
def count_images_in_subfolders(path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    result_dict = {}

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            image_count = 0
            for root, dirs, files in os.walk(subdir_path):
                for file in files:
                    if any(file.endswith(ext) for ext in image_extensions):
                        image_count += 1
            result_dict[subdir] = image_count

    return result_dict   
import os
train_stats=count_images_in_subfolders(os.path.join(data_root, "train"))
test_stats=count_images_in_subfolders(os.path.join(data_root, "test"))
classes=extract_category_info(os.path.join(data_root, "train"))
print("classes",classes)
classes.sort()
print("classes after sorting",classes)
#classes=count_classes(train_ann_file)

num_classes = len(classes)
print(f"num_classes:{num_classes}")
metainfo = dict(classes=classes, )
print(f"metainfo {metainfo}")
top_k_second=5 if num_classes > 5 else num_classes
model = dict(
    type='ImageClassifier',
    head=dict(
        num_classes=num_classes,
       # topk=(1, 5) 
        ),
       
    _scope_='mmpretrain')
# dataset settings


dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True,
)
policies = [
    dict(type='AutoContrast', prob=0.5),
    dict(type='Equalize', prob=0.5),
    dict(type='Invert', prob=0.5),
    dict(
        type='Rotate',
        magnitude_key='angle',
        magnitude_range=(0, 30),
        pad_val=0,
        prob=0.5,
        random_negative_prob=0.5),
    dict(
        type='Posterize',
        magnitude_key='bits',
        magnitude_range=(0, 4),
        prob=0.5),
    dict(
        type='Solarize',
        magnitude_key='thr',
        magnitude_range=(0, 256),
        prob=0.5),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110),
        thr=128,
        prob=0.5),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Contrast',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Brightness',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Sharpness',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='horizontal',
        random_negative_prob=0.5),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='vertical',
        random_negative_prob=0.5),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='horizontal',
        random_negative_prob=0.5,
        interpolation='bicubic'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='vertical',
        random_negative_prob=0.5,
        interpolation='bicubic')
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=image_size, edge='short', backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandAugment',
        policies=policies,
        num_policies=8,
        magnitude_level=12),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=image_size, edge='short', backend='pillow'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='',
        data_prefix='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1,))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator


default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5)
    )
import platform
wandb_name=data_root.split("/")[-1]
if(wandb_name==""):
    wandb_name=data_root.split("/")[-2]
wandb_name=f"dataset{wandb_name}-batchsize_{batch_size}-maxep_{train_max_epochs}-{platform.node()}"
plan_name=f"{_base_.model.backbone.type}-image_size_{image_size}_batch_size{batch_size}-{wandb_name}"
visualizer = dict(
    vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'), 
        dict(type='WandbVisBackend',
        init_kwargs={'project': f'palmdate-cls','name':plan_name},)
    ]) # noqa
import datetime    

work_src=f"./work_dirs/palmdate_cls_{plan_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
work_dir=f"{work_src}/out/"