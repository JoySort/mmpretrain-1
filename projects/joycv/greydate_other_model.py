_base_ = 'mmpretrain::repvgg/repvgg-A0_8xb32_in1k.py' #1.36 - 72.37
#_base_ = 'mmpretrain::levit/levit-256_8xb256_in1k.py' #1.14 - 81.59
#_base_ = 'mmpretrain::levit/levit-192_8xb256_in1k.py' #0.67 - 79.86
#_base_ = 'mmpretrain::mobilevit/mobilevit-small_8xb128_in1k.py' #2.03 - 78.25
#_base_ = 'mmpretrain::mobilenet_v3/mobilenet-v3-large_8xb128_in1k.py' # 0.23 - 74.04



train_max_epochs=1000


#关于batch size的使用原则：
#Batch size是指每次训练所用的样本数，一般情况下，batch size越大，训练速度越快，但是对于内存和显存的需求也越高，因此需要根据具体情况进行选择。
#当样本数量较少时，可以适当增大batch size，以充分利用计算资源，提高训练效率。但是当样本数量较大时，可以选择较小的batch size，以避免内存和显存不足导致训练失败。
#一般经验，针对repvgg模型，8G 显存224图片可以支持到128，12G显存224图片可以支持到256。简单的
#另外，较大的batch size可能会导致过拟合，因此需要在训练过程中观察训练集和验证集的loss和accuracy，及时调整batch size大小，避免过拟合的发生。
batch_size=256 
image_size=256
data_root="/nas/win_essd/UAE_sliced_256/pd_train_candidate/intermediate_model_candidate/"
data_root="/nas/win_essd/西梅/训练样本/"
data_root="/opt/workspace/imagedb/yan/"
data_root="/nas/win_essd/UAE_sliced_256/pd_train_candidate/intermediate_model_candidate/"

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
    head=dict(
        num_classes=num_classes,
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
        init_kwargs={'project': f'greydate_mmpretrain-cls','name':f"{_base_.model.backbone.type}-{wandb_name}"},)
    ]) # noqa