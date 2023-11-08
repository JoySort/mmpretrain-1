import os
import random

import shutil
from pathlib import Path
from random import sample

# Set path to parent folder containing class folders
parent_folder = '/nas/win_essd/khalas_calxy_training/train3/train/' 

# Get paths for the two class folders
class1_folder = os.path.join(parent_folder, 'calxy')
class2_folder = os.path.join(parent_folder, 'none_calxy')

# Count files for each class
class1_count = sum([len(files) for r, d, files in os.walk(class1_folder)])
class2_count = sum([len(files) for r, d, files in os.walk(class2_folder)])

# Determine difference in count
diff = abs(class1_count - class2_count)

# Find class with more samples
if class1_count > class2_count:
    higher = class1_folder
    lower = class2_folder
else:
    higher = class2_folder
    lower = class1_folder
    
print(f"higher class is {higher} with count \r\n{class1_folder} {class1_count}  \r\n{class2_folder}{class2_count}")    
print(f"remove parameters is {higher} with diff{diff}")
# Pick random files to remove from higher class
#del_files = sample(os.listdir(higher), k=diff) 
del_files =random.sample(os.listdir(higher), diff)
# Delete those files
for file in del_files:
    os.remove(os.path.join(higher, file))
    
print("Classes balanced by deleting", diff, "files from", higher)