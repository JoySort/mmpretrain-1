import argparse
import cv2
import os
import sys

parser = argparse.ArgumentParser(description='BMP to JPEG converter')
parser.add_argument('src_folder', help='Source folder path')
parser.add_argument('dest_folder', help='Destination folder path')
args = parser.parse_args()

src_folder = args.src_folder
dest_folder = args.dest_folder 

if not os.path.isdir(src_folder):
    print("Source folder not found!")
    sys.exit()
if not os.path.exists(dest_folder):
   print(f"Destination {dest_folder} does not exist. Creating it.")
   os.makedirs(dest_folder)
   
for root, dirs, files in os.walk(src_folder):

    for filename in files:

        if filename.endswith(".bmp"):

            filepath = os.path.join(root, filename)  
            img = cv2.imread(filepath)

            relative_path = os.path.relpath(root, src_folder)
            out_path = os.path.join(dest_folder, relative_path)
            os.makedirs(out_path, exist_ok=True)

            clean_name = os.path.splitext(filename)[0]
            out_file = os.path.join(out_path, clean_name + ".jpg")

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(out_file, img) 

            print(f"Converted {filename} to {out_file}")
            
print("Conversion complete!")