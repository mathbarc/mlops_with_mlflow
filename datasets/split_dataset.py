import os
import pathlib
import random
import shutil


base_dataset_dir = "./Rice_Image_Dataset/"

train_dataset_dir = os.path.join(base_dataset_dir, "train")
test_dataset_dir = os.path.join(base_dataset_dir, "test")

if not os.path.exists(train_dataset_dir):
    os.makedirs(train_dataset_dir)

if not os.path.exists(test_dataset_dir):
    os.makedirs(test_dataset_dir)

train_test_split_ratio = 0.8

label_counter = {}

for file_path in pathlib.Path(base_dataset_dir).rglob("*.jpg"):

    parent_folder = os.path.dirname(file_path)
    label = os.path.basename(parent_folder)

    if label not in label_counter:
        label_counter[label] = 0
        os.makedirs(os.path.join(train_dataset_dir,label))
        os.makedirs(os.path.join(test_dataset_dir,label))
    
    if (random.randint(0,100)/100) < train_test_split_ratio:
        dest = os.path.join(train_dataset_dir,label,f"{label_counter[label]}_{label}.jpg")
    else:
        dest = os.path.join(test_dataset_dir,label,f"{label_counter[label]}_{label}.jpg")
    
    shutil.copy(file_path, dest)
    label_counter[label]+=1



