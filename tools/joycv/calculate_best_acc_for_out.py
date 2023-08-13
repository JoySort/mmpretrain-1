import os
import json
import re

work_dir = "./work_dirs/chestnut_core_repvgg"
log_file_pattern = "log.json"

pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')
files = os.listdir(work_dir)
train_paths = [file for file in files if pattern.match(file)]


max_accuracy = -1
best_epoch = None
best_train_path = None

info_names=[]
# loop through each subdirectory
for train_path in train_paths:
  #print(train_path)
  # get the full path to the subdirectory
  log_path = os.path.join(work_dir, train_path,'out')
  #print(log_path)
  if not os.path.exists(log_path):
    info_obj={'path':train_path,'name':f"unfinished_training"}
    info_names.append(info_obj)
    continue
  model_files = [f for f in os.listdir(log_path) if f.endswith("pth")]
  
  # Remove '.pth' extension and extract only digits from the file name
  int_list = [ ]
  for file_name in model_files:
    if(file_name.startswith("epoch")):
        int_list.append(int(file_name.split(".")[0].split("_")[-1]))
  # Sort the list of integers
  int_list.sort()
  if(len(int_list)>0):
    # Find the maximum value
    max_value = max(int_list)
    #for filename in int_list:

    print(f"Max value: {max_value} from {log_path}")
  else:
    print(f"empty list from {log_path}")

  # get a list of all the files in the subdirectory that match the log file pattern
  log_files = [f for f in os.listdir(log_path) if f.endswith(log_file_pattern)]
  
#   for f in os.listdir(log_path):
#     print(f)
  # loop through each log file
  for log_file in log_files:
    # get the full path to the log file
    log_file = os.path.join(log_path,log_file)
    #print(log_file)
    # Open the file for reading
    with open(log_file, 'r') as file:
        # Initialize a variable to store the highest value from the "accuracy_top-1" key
        max_accuracy = 0

        # Loop through each line in the file
        for line in file:
            # Try to parse the line as a JSON object
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # If the line is not a valid JSON object, skip to the next line
                continue
            
        # If the "accuracy_top-1" key exists in the JSON object, compare its value to the current max_accuracy
        if "accuracy_top-1" in obj and obj["accuracy_top-1"] > max_accuracy:
            # If the value is greater, update the max_accuracy
            max_accuracy = obj["accuracy_top-1"]
            epoch=obj["epoch"]
            info_obj={'path':train_path,'name':f"best_acc-{max_accuracy}-epoch{epoch}"}
            info_names.append(info_obj)
                #print(info_obj)
        # After looping through all the lines, print the max_accuracy
    #for f in os.listdir(log_path):
for train_path in train_paths:
  #print(train_path)
  # get the full path to the subdirectory
  log_path = os.path.join(work_dir, train_path)    
  print(train_path)
  for info_item in info_names:
    if (info_item['path']==train_path):
      dest_dir=os.path.join(work_dir, f"{train_path}_{info_item['name']}")  
      print(f"move from {log_path} to {dest_dir}")
      #os.rename(log_path,dest_dir)
  
#print("info_names:", info_names)
