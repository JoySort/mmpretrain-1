import os
import sys

if len(sys.argv) < 2:
    print('Error: start path not provided')
    sys.exit(1)

start_path = sys.argv[1]

print(f'Starting rename for {start_path}')

input("Press Enter to continue...")

for root, dirs, files in os.walk(start_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.bmp', '.png')):
            
            if '[' in file:
                print(f'Renaming file {file}')
                new_name = file[:file.index('[')]
                os.rename(os.path.join(root, file), os.path.join(root, new_name))

print('Done!')