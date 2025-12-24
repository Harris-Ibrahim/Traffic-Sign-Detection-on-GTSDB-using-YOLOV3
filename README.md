# Traffic Sign Detection on GTSDB using YOLOV3
Traffic signs encode important information about road conditions. Detecting traffic signs is therefore a critical component of modern Advanced Driver Assistance Systems (ADAS) and Autonomous Driving. These detections must be performed in all weather and road conditions with high precision, high recall and in real time. In this project, we develop a 2D traffic sign detector by adapting the YOLO-v3 object detector on the German Traffic Sign Detection Benchmark. We achieved a 91 percent mean Average Precision (mAP) after finetuning the YOLO-v3 model previously trained on the MS COCO dataset when detecting traffic signs of 4 general categories. We acheived a 14 percent mAP when detecting all 43 classes. Improvements to the training process, model architecture and dataset are required to achieve better results. 

https://github.com/user-attachments/assets/909898f6-8856-4efa-9088-5e71cc7ed759


## Methods
### Dataset : German Traffic Sign Detection Benchmark (GTSDB) [2]
The dataset consists of 900 images containing road scenes with traffic signs. Each image includes one or more instances of traffic signs in a diverse range of weather, lighting, occlusion conditions and distances to the camera. The first 600 images are reserved for training, the later 300 are for testing. The annotations consist of 2D bounding box coordinates of each traffic sign in the scene with the corresponding class. There are a total of 43 classes of traffic signs. The original benchmark groups these 43 classes into 4 superclasses. The following image is from [6].![Dataset Categories](assets/traffic_sign_catogries_gtsdb.png)

We train to predict only among these 4 superclasses. Following are some sample images from the dataset along with the bounding box, class labels and a dummy confidence score. The dataset format is converted to a compatible form. ![Dataset Samples](assets/dataset_samples.png)

### Model Architecture
YOLO-V3 [3] is a general purpose object detector. The backbone of the network is a 53-layer CNN named Darknet-53. The extracted features of the backbone are then passed to the detection heads at three different scales. This is inspired by Feature Pyramid Networks. The three detection heads generate bounding box predictions at three different resolution. We made no changes to the architecture and used a Pytorch implementention by [1]

Our motivations for using YOLO-v3 instead of later versions are that it is fast at inference time, relatively simple to implement from scratch and requires less compute at training time. Due to the limitations of the dataset, GPU compute and time, we use YOLO-v3. 

### Evaluation Criteria
In scientific literature on traffic sign detection and generally object detection, mean Average Precision (mAP) is used to evaluate the performance of object detectors. We used the mean Average Precision @ 50 criteria as used in Pascal VOC 2007 dataset

### Data Augmentation
 We used data augmentation strategies like random crop, colour jitter (randomly change brightness, contrast, hue, saturation), affine transformation (change zoom level, translate left and right, rotate, and shear), horizontal flip, random blurring, colour channel shuffling, and grey scale. All these were applied with various probabilities. Finally, the images are standardized (mean 0, std 1). These transforms are also applied to the bounding box labels. 

 The images are also resized to 608x608 with a padding of 1 to reduce distortion.

### Anchor boxes
YOLO-v3 like many other object detectors does not predict the absolute coordinates of the bounding box. The bounding box centre coordinates are predicted as offsets to the position of the top left corner of the grid cell. The width and height of the box are predicted as an offset of the width and height of anchor boxes.

Anchor boxes should be chosen based on the likely size of the bounding boxes for the particular problem at hand. The authors of YOLO-v3 recommend clustering analysis on the ground truth bounding boxes to choose the best sizes for anchor boxes.

### Training
We used the pre-trained weights of YOLO-V3 MS-COCO from the orignal authors. These were obtained by first training the backbone of the model on the ImageNet classification dataset. The authors then trained the backbone along with the detection heads on MS-COCO.
After initializing a model with these pretrained weights, we train for 100 epochs on the training set of GTSDB on a learning rate of 1e4. After thaat, we train for a further 100 epochs using a learning rate of 1e5. We then test the performance on the test set

## Results
### Average Precision Per Class
Precision and recall curves for each class are calculated. The area under the curve gives us the Average Precision (AP) for each class. The AP for each class is reasonablly good. The trained model performs the worst on "Other". This class consists of an assortment of traffic signs that cannot be neatly grouped in the rest of categories. Many of these only have only a dozen or less samples in the dataset. ![Average Precision per Class](assets/average_precision_bar_chart_page.jpg)

### Overall Mean Average Precision (mAP)
We obtained a mAP of 91 percent while detecting 04 classes with only 600 training images. The mAP drops to just 11 percent while detecting all 43 classes. This is likely because some classes have a dozen or less samples

| Model                   | Problem                     | mAP @ 50 IoU |
| ----------------------- |:---------------------------:|:------------:|
| YOLOv3 	                | GTSDB (4 Super classes only)| 91           |
| YOLOv3                  | GTSDB (All 43 Classes)      | 11           |

### Effect of Better Anchor boxes:
Initially, we used the anchor box dimensions obtained by clustering analysis on MS-COCO as reported by the Authors of YOLO-V3. We then use the anchor box dimensions reported by [5] who did clustering analysis on GTSDB and reported the recommended anchor box dimensions. We observed that using better anchors, while keeping everything else constant, improved training and resulted in a **3 percent increase in mAP**.

### Sample Predictions:
Even with only 600 training images, the model is very good at correctly localising and detecting most traffic signs. A weakness is correctly classifying some signs, especially those belonging to the "Other" class. ![Sample Predictions](assets/model_predictions.png)

### Common Mistakes:
Just like other traffic sign detector, a common problem is incorrectly detecting many common road objects as traffic signs. For example, some car tail lights are detected as traffic signs. Makeshift road barriers with red and white stripes are also incorrectly detected. Finally solar glare and reflections are also many time incorrectly detected as a traffic sign. Some traffic signs, such as advertisements and bus stop signs are also confused with traffic signs. ![Incorrect Predictions](assets/model_mistakes.png)

### Comparison with the State of the Art
There are State of the Art architectures which have 09 percent better mAP precision. Still YOLO-V3 has above 90 percent mAP while requiring far less compute at test and training time than many competing methods.

| Paper | Architecture | Mean Average Precision |
|------|-------------|------------------------|
| Zheng et al. [7] | Vision Transformer | 98.7 |
| Hamed Aghdam et al. [8] | Sliding Window CNN | 99.8 |
| Alvaro Arcos-Garcia et al. [9] | Faster R-CNN | 95.8 |
| Yawar Rehman et al. [10] | Custom YOLO-V3 | 93.1 |
| **This work** | Base YOLO-V3 | 91.1 |

## Reproducing Results
### Setting up environment
- Set up a Python 3.12 environment. Python 3.12 [Miniconda Distribution](https://www.anaconda.com/download)
- Install the required packages by using pip and the provided requirements.txt file
- Install Pytorch with Cuda 13.0 using the following command:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
- In case, there are problems with installing or running Pytorch or you want some other version of Pytorch then check their [instructions](https://pytorch.org/get-started/locally/).

### GTSDB Dataset
- Download the full GTSDB Dataset file named "FullIJCNN2013.zip" from [link](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html)
- Extract the FullIJCNN2013.zip to the root directory so that the structure is root_dir / FullIJCNN2013 / FullIJCNN2013 / Dataset files

### Download original YOLO-V3 Weights 
Download YOLOv3-608 weights for MS-COCO Dataset from [Link](https://data.pjreddie.com/files/yolov3.weights) and move the file to the root_dir / checkpoints. Save the weights to PyTorch format by running YOLOV3/model_with_weights.py.

### Convert GTSDB to a Compatible Format
Run the script file Main / GTSDB_Conversion.py to convert the GTSDB dataset files into a format compatible with the project

### Training and Evaluation
- Run the notebook Main / Main_GTSDB_V3.ipynb to create datasets, dataloaders, model objects and then train them and evaluate them on the dataset
- Edit the YOLOV3 / config.py to change experimental setup as necessary

## Built With
Python, Pytorch

## References
1. Demo video has been created by generating predictions on high quality raw footage by 

1. German Traffic Sign Detection Benchmark :
@inproceedings{Houben-IJCNN-2013,
   author = {Sebastian Houben and Johannes Stallkamp and Jan Salmen and Marc Schlipsing and Christian Igel},
   booktitle = {International Joint Conference on Neural Networks},
   title = {Detection of Traffic Signs in Real-World Images: The {G}erman {T}raffic {S}ign {D}etection {B}enchmark},
   number = {1288},
   year = {2013},
}
1. Yolov3 is an object detector based on the following research paper:
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
1. This repository uses code adapted from the [Pytorch implementation of YOLOV3](https://github.com/SannaPersson/YOLOv3-PyTorch) by Aladin Persson and Sana Persson 

1. Anchor box sizes for YOLO-V3 suitable for GTSDB by doing clustering analysis on the bounding box labels:
@article{yolov3_better_anchors,
    doi = {10.1088/1757-899X/787/1/012034},
    year = {2020},
    month = {mar},
    publisher = {IOP Publishing},
    volume = {787},
    number = {1},
    pages = {012034},
    author = {Liu, XiongFei and Xiong, Fan},
    title = {A Real-time Traffic Sign Detection Model Based on Improved YOLOv3},
    journal = {IOP Conference Series: Materials Science and Engineering},
    }

1. GTSDB Superclasses 
@Article{yolov4_gtsdb,
AUTHOR = {Gu, Yang and Si, Bingfeng},
TITLE = {A Novel Lightweight Real-Time Traffic Sign Detection Integration Framework Based on YOLOv4},
JOURNAL = {Entropy},
VOLUME = {24},
YEAR = {2022},
NUMBER = {4},
ARTICLE-NUMBER = {487},
DOI = {10.3390/e24040487},
}

1. Vision Transformer
@article{vision_transformer,
author = {Zheng, Yuping and Jiang, Weiwei},
title = {Evaluation of Vision Transformers for Traffic Sign Classification},
journal = {Wireless Communications and Mobile Computing},
volume = {2022},
number = {1},
pages = {3041117},
doi = {https://doi.org/10.1155/2022/3041117},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1155/2022/3041117},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1155/2022/3041117},
year = {2022}
}

1. Sliding Window CNN
@article{gtsdb_rank2_sliding_window_CNN,
title = {A practical approach for detection and classification of traffic signs using Convolutional Neural Networks},
journal = {Robotics and Autonomous Systems},
volume = {84},
pages = {97-112},
year = {2016},
issn = {0921-8890},
doi = {https://doi.org/10.1016/j.robot.2016.07.003},
url = {https://www.sciencedirect.com/science/article/pii/S092188901530316X},
author = {Hamed {Habibi Aghdam} and Elnaz {Jahani Heravi} and Domenec Puig},
keywords = {Convolutional Neural Networks, Traffic sign detection, Traffic sign classification, Sliding window detection, Dense prediction}
}

1. Fast RCNN
@article{gtsdb_rank1_fasterrcnn,
title = {Evaluation of deep neural networks for traffic sign detection systems},
journal = {Neurocomputing},
volume = {316},
pages = {332-344},
year = {2018},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2018.08.009},
url = {https://www.sciencedirect.com/science/article/pii/S092523121830924X},
author = {Álvaro Arcos-García and Juan A. Álvarez-García and Luis M. Soria-Morillo},}
}

1. Custom YOLOV3
@Article{gtsdb_rank2_Custom_yolov3,
AUTHOR = {Rehman, Yawar and Amanullah, Hafsa and Saqib Bhatti, Dost Muhammad and Toor, Waqas Tariq and Ahmad, Muhammad and Mazzara, Manuel},
TITLE = {Detection of Small Size Traffic Signs Using Regressive Anchor Box Selection and DBL Layer Tweaking in YOLOv3},
JOURNAL = {Applied Sciences},
VOLUME = {11},
YEAR = {2021},
NUMBER = {23},
ARTICLE-NUMBER = {11555},
URL = {https://www.mdpi.com/2076-3417/11/23/11555},
ISSN = {2076-3417},
DOI = {10.3390/app112311555}
}

