import os


folder_name="worm\\first"
src_path=os.path.join("C:\\opt\\workspace\\imagedb\\",folder_name)
train_path=os.path.join("C:\\opt\\workspace\\imagedb\\freshchestnut\\train",folder_name)
test_path=os.path.join("C:\\opt\\workspace\\imagedb\\freshchestnut\\test",folder_name)
manual_path=os.path.join("C:\\opt\\workspace\\imagedb\\freshchestnut\\manual",folder_name)

os.makedirs(train_path,exist_ok=True)
os.makedirs(test_path,exist_ok=True)
os.makedirs(manual_path,exist_ok=True)

import glob
img_file_list=glob.glob(f"{src_path}/*.bmp")

import random
random.shuffle(img_file_list)

####################################################################
count=len(img_file_list)
print(count)

rtrain=6
rtest=1
rmanual=0
rtotal=rtrain+rtest+rmanual

num_test=int(count*rtest/rtotal)
num_manual=int(count*rmanual/rtotal)
num_train=count-num_test-num_manual

print(num_train)
print(num_test)
print(num_manual)

import shutil
count=0
for img_file in img_file_list:
    count=count+1
    if(count<=num_train):
        shutil.copy(img_file,train_path)
    elif(count<=num_test+num_train):
        shutil.copy(img_file,test_path)
    else:
        shutil.copy(img_file,manual_path)




#print(count)

