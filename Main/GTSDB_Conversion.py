''' 
Convert the Format and structure of the dataset

We will change the structure: 
    Create a new directory with a dir for images and a dir for gt.txt files. 
    Images dir contains all images in jpg, labels dir contains .txt files with labels. 
    There is a csv file matching the id of each image with the id of its label txt file 

Current label format: 
    X_min, Y_min, X_max, Y_max. 
    Where X is along the height axis of the image and Y is along the width axis of the image. These coordinates are also absolute pixel positions. 

We will change them into the format:
    X_centre, Y_centre, Width, Height. 
    Where X_centre is the centre coordinate of the box along the width axis, Y_centre is the centre coordinate of the box along the height axis. 
    We will also normalize them with image size. So they are between 0-1. Where 1 corresponds to the ends of the image and zero to the start.

We will change the Class id to reduce classes:
    from class id in 0-42 range to the corresponding 0-3 super class
'''

from pathlib import Path
import os
import pandas as pd      
from PIL import Image
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

root_dir = Path(root_dir)

# List to classes that belong to each super class
OTHER_CLASSES = [6, 12, 13, 14, 17, 32, 41, 42]
PROHIBITORY_CLASSES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
MANDATORY_CLASSES = [33, 34, 35, 36, 37, 38, 39, 40]
DANGER_CLASSES = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

# Take one of the 43 classes and return the corresponding super class it belongs to
def map_to_superclass (cls):
    if cls in OTHER_CLASSES:
        return 0
    elif cls in PROHIBITORY_CLASSES:
        return 1
    elif cls in MANDATORY_CLASSES:
        return 2
    elif cls in DANGER_CLASSES:
        return 3
    else:
        raise ValueError("Class idx is beyond [0-42]")

# Source dirs
dataset_dir = root_dir/"FullIJCNN2013"/"FullIJCNN2013"

# Target dirs
target_dir = root_dir/ "GTSDB_YOLO"
images_dir = target_dir / "images"
labels_dir = target_dir / "labels"

# Create target dir
target_dir.mkdir(parents= True, exist_ok= True)
images_dir.mkdir(parents= True, exist_ok= True)
labels_dir.mkdir(parents= True, exist_ok= True)

all_csv_file_path = target_dir / "all.csv"
train_csv_file_path = target_dir / "train.csv"
test_csv_file_path = target_dir / "test.csv"

gt_lines = []
gt_path = dataset_dir / "gt.txt"

print ("Starting Conversion... \n")

# Open the gt.txt file, append each line to the list
if gt_path.exists():
    with open(gt_path) as f:
        for line in f:
            gt_lines.append((line.strip()))


# Process each line in gt.txt and add them to a new dictionary where each image idx contains a list of all the gt labels for that image id
# so the structure is a list of dictionaries, where each dict has key: image path for that specific image, value: List of tuples for gt for that image
from collections import defaultdict

annotations = defaultdict(list)
for line in gt_lines:
    filename, x1, y1, x2, y2, class_id = line.split(';')
    image_path = dataset_dir / filename
    annotations[image_path].append((int(class_id), int(x1), int(y1), int(x2), int(y2)))

annotations_csv_data = []

# iterate through all the images and their gt in annotations dict
for image_path, boxes in annotations.items():

    # Open image, get size, change format and save
    with Image.open(image_path) as img:
        # Get image size for changing scale of labels later on
        img_width, img_height  = img.size
        # Convert image to jpg and copy
        target_img_path = images_dir / (image_path.stem + ".jpg")
        img.convert("RGB").save(target_img_path)

        # Convert the class_id to the corresponding super class id
        # Convert the format of label coordinates to YOLO format
        label_lines = []
        for class_id, x1, y1, x2, y2 in boxes:
            super_class_id = map_to_superclass(class_id)

            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            box_width = (x2 - x1) / img_width
            box_height = (y2 - y1) / img_height

            label_lines.append(f"{super_class_id} {x_center} {y_center} {box_width} {box_height}")

        # Save label file
        label_file = labels_dir / (image_path.stem + ".txt")
        with open(label_file, 'w') as f:
            f.write('\n'.join(label_lines))

        # Add to CSV data
        annotations_csv_data.append((image_path.stem, target_img_path.name, label_file.name))

# Create a annotations df which matches image_id to the label_id
annot_df = pd.DataFrame(annotations_csv_data, columns=["image_idx", "image_id", "label_id"])
annot_df.to_csv(all_csv_file_path, index=False)

# Reload the csv
annot_df = pd.read_csv (filepath_or_buffer= all_csv_file_path)

# All the images with id < 600 are to be used for training. Their paths will now be stored in a separate csv
# image_idx col is just image id without .jpg extension
train_df = annot_df.where(cond= annot_df["image_idx"] < 600)
train_df = train_df.drop (columns= "image_idx")
train_df = train_df.dropna()

train_df.to_csv(train_csv_file_path, index=False)

# All the images with id >= 600 are to be used for testing. Their paths will now be stored in a separate csv
test_df = annot_df.where(cond= annot_df["image_idx"] >= 600)
test_df = test_df.drop (columns= "image_idx")
test_df = test_df.dropna()

test_df.to_csv(test_csv_file_path, index=False)

# Also save the csv with all the image and label ids
annot_df = annot_df.drop (columns= "image_idx")
annot_df.to_csv(all_csv_file_path, index=False)

print ("Completed Conversion \n")
