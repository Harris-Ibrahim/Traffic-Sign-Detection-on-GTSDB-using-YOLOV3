# Traffic Sign Detection on GTSDB using YOLOV3
Traffic signs encode important information about road conditions. Detecting traffic signs is therefore a critical component of modern Advanced Driver Assistance Systems (ADAS) and Autonomous Driving. These detections must be performed in all weather and road conditions with high precision, high recall and in real time. In this project, we develop a 2D traffic sign detector by adapting the YOLO-v3 object detector on the German Traffic Sign Detection Benchmark. We achieved a 91 percent mean Average Precision (mAP) after finetuning the YOLO-v3 model previously trained on the MS COCO dataset when detecting traffic signs of 4 general categories. We acheived a 14 percent mAP when detecting all 43 classes. Improvements to the training process, model architecture and dataset are required to achieve better results. 


## Dataset : German Traffic Sign Detection Benchmark (GTSDB) [1]





## Setting up environment
- Set up a Python 3.12 environment. Python 3.12[Miniconda Distribution](https://www.anaconda.com/download)
- Install the required packages by using pip and the provided requirements.txt file
- Install Pytorch with Cuda 13.0 using the following command:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
- In case, there are problems with installing or running Pytorch then check their [instructions](https://pytorch.org/get-started/locally/).

## GTSDB Dataset
- Download the full GTSDB Dataset file named "FullIJCNN2013.zip" from [link](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html)
- Extract the FullIJCNN2013.zip to the root directory so that the structure is root_dir / FullIJCNN2013 / FullIJCNN2013 / Dataset files

## Download original YOLO-V3 Weights 
Download YOLOv3-608 weights for MS-COCO Dataset from [Link](https://data.pjreddie.com/files/yolov3.weights) and move the file to the root_dir / checkpoints. Save the weights to PyTorch format by running YOLOV3/model_with_weights.py.

## Built With
Python, Pytorch

## Acknowledgements
This repository uses code adapted from the [Pytorch implementation of YOLOV3](https://github.com/SannaPersson/YOLOv3-PyTorch) by Aladin and Sana Persson 

Yolov3 is an object detector based on the following research paper:
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}