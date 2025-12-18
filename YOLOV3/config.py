import albumentations as A
import cv2
import torch
import sys
import os
from albumentations.pytorch import ToTensorV2
from pathlib import Path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
root_dir = Path (root_dir)

DETECTION_DATASET = 'GTSDB_YOLO'
RECOGNITION_DATASET = 'GTSRB_YOLO'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 1
BATCH_SIZE = 16
IMAGE_SIZE = 608
NUM_CLASSES = 4
# Schedule : 
# freeze 1-11: 0 - 30 epochs LEARNING_RATE = 1e-3   
# unfreeze all: 30 - 50 epochs LEARNING_RATE = 1e-4
# unfreeze all: 50 - 70 epochs LEARNING_RATE = 1e-5
LEARNING_RATE = 1e-5   
WEIGHT_DECAY = 1e-4
NUM_EPOCHS =  30
CONF_THRESHOLD = 0.6
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = os.path.join(root_dir, "checkpoints//checkpointv3.pth.tar")

ANCHORS = [
    [(0.191, 0.148), (0.257, 0.326), (0.613, 0.536)],
    [(0.049, 0.100), (0.102, 0.074), (0.097, 0.196)],
    [(0.016, 0.021), (0.026, 0.049), (0.054, 0.039)],
]  # Note these have been rescaled to be between [0, 1]

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),

        A.Affine(
            # Tuple for scale (will be used for both x and y)
            scale=(0.8, 1.2),
            # Dictionary with tuples for different x/y translations
            translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)},
            # Tuple for rotation range
            rotate=(-30, 30),
            # Dictionary with tuples for different x/y shearing
            shear={"x": (-10, 10), "y": (-5, 5)},
            # Interpolation methods
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            # Other parameters
            fit_output=False,
            keep_ratio=True,
            rotate_method="largest_box",
            balanced_scale=True,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0, 
            p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.MotionBlur(p = 0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
        ],
bbox_params=A.BboxParams(format="yolo", min_visibility=0.3, label_fields=[],)
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.3, label_fields=[]),
)

Reduced_GTSDB_Classes = [
    "Other",
    "Prohibitory",
    "Mandatory",
    "Danger"
 ]

GTSDB_Classes = [
    "speed limit 20",
    "speed limit 30",
    "speed limit 50",
    "speed limit 60",
    "speed limit 70",
    "speed limit 80",
    "restriction ends 80",
    "speed limit 100",
    "speed limit 120",
    "no overtaking",
    "no overtaking (trucks)",
    "priority at next intersection ",
    "priority road",
    "give way",
    "stop",
    "no traffic both ways",
    "no trucks",
    "no entry",
    "danger",
    "bend left",
    "bend right",
    "bend",
    "uneven road",
    "slippery road",
    "road narrows",
    "construction",
    "traffic signal",
    "pedestrian crossing",
    "school crossing",
    "cycles crossing",
    "snow",
    "animals",
    "restriction ends",
    "go right",
    "go left",
    "go straight",
    "go right or straight",
    "go left or straight",
    "keep right",
    "keep left",
    "roundabout",
    "restriction ends",
    "restriction ends (trucks)",
]

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]