from mmcls.apis import init_model, inference_model
import mmcv
import time,os
import json
import numpy as np
import shutil
import datetime
from mmcls.apis.inference_torch import inference_torch
import cv2

import torch



# optionally set the print options to disable scientific notation
torch.set_printoptions(sci_mode=False)


def inference_with_torch():
    image = cv2.imread("/opt/workspace/imagedb/chestnut_core/inference_result/epoch_1000__opt_workspace_joycpplib_cmake-build-debug_runner_app_debug_images_/2023-03-28_08-54-12/tiny_black/rot/tiny_black-0.99991-rot0.00007-chestnut-core-segmentation-validation_png_4_sliced_image_r2_c3.jpg")

    checkpoint="/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-22_22-49-57/out/epoch_1000_[2023_03_26]-[02_26_34]/torchscript.pt"
    config="/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-22_22-49-57/out/epoch_1000_[2023_03_26]-[02_26_34]/torchscript.json"
    config_py="/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-22_22-49-57/out/chestnut_core_repvgg.py"
    checkpoint_py="/opt/workspace/mmcls_gitee/work_dirs/chestnut_core_repvgg/2023-03-22_22-49-57/out/epoch_1000.pth"
    model = init_model(config_py, checkpoint_py, device='cuda:0')
    result2,score = inference_model(model,image )
    print(result2,score)
   
    print(inference_torch(checkpoint,config,image))


inference_with_torch()