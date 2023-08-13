_base_ = 'mmpretrain::repvgg/repvgg-A0_8xb32_in1k.py'
#_base_ = 'mmpretrain::levit/levit-256_8xb256_in1k.py'
#
train_max_epochs=1000
#训练微调参数
#关于 dropout的使用原则： 
#Dropout是一种用于防止过拟合的技术，可以随机丢弃神经元的输出，以减少神经元之间的依赖关系，促进网络泛化能力的提升。
#Dropout的取值范围在0到1之间，代表着丢弃神经元的比例。一般来说，0.5是一个较好的起点，可以逐步增加或减少，观察训练效果，选择最优值。
#当网络较小时，可以采用较小的dropout，避免过度减少网络容量，导致欠拟合。而当网络较大时，可以采用较大的dropout，以增加网络的泛化能力，防止过拟合。
dropout=0.7

#关于batch size的使用原则：
#Batch size是指每次训练所用的样本数，一般情况下，batch size越大，训练速度越快，但是对于内存和显存的需求也越高，因此需要根据具体情况进行选择。
#当样本数量较少时，可以适当增大batch size，以充分利用计算资源，提高训练效率。但是当样本数量较大时，可以选择较小的batch size，以避免内存和显存不足导致训练失败。
#一般经验，针对repvgg模型，8G 显存224图片可以支持到128，12G显存224图片可以支持到256。简单的
#另外，较大的batch size可能会导致过拟合，因此需要在训练过程中观察训练集和验证集的loss和accuracy，及时调整batch size大小，避免过拟合的发生。
batch_size=256 

data_root="/opt/workspace/imagedb/chestnut_core_sliced/0616_revision/"

checkpoint_config = dict(interval=25)
train_cfg = dict(by_epoch=True, max_epochs=train_max_epochs, val_interval=1)



#data_root='/opt/images/UAE/pd_train_candidate/intermediate_model_candidate/'





def extract_category_info(data_root):
    import os

    subfolder_names = []
    
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        
        if os.path.isdir(item_path):  # Check if it's a directory
            subfolder_names.append(item)
    subfolder_names.sort()
    return subfolder_names
   

classes=extract_category_info(data_root+"train")
print("classes",classes)
classes.sort()
print("classes",classes)
#classes=count_classes(train_ann_file)

num_classes = len(classes)
print(f"num_classes:{num_classes}")
metainfo = dict(classes=classes, )
print(f"metainfo {metainfo}")

model = dict(
    type='ImageClassifier',
    backbone=dict(type='RepVGG', arch='A0', out_indices=(3, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)),
    _scope_='mmpretrain')
# dataset settings

image_size=224
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
    dict(type='RandomResizedCrop', scale=image_size, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
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
    dict(type='CenterCrop', crop_size=image_size),
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
    checkpoint=dict(interval=10, max_keep_ckpts=3, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5)
    )
import platform
wandb_name=data_root.split("/")[-1]
if(wandb_name==""):
    wandb_name=data_root.split("/")[-2]
wandb_name=f"dataset{wandb_name}-batchsize_{batch_size}-maxep_{train_max_epochs}-{platform.node()}"
visualizer = dict(
    vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'), 
        dict(type='WandbVisBackend',
        init_kwargs={'project': f'chestnut_mmpretrain-cls','name':f"{wandb_name}"},)
    ]) # noqa