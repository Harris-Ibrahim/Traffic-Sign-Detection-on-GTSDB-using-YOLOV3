## Setting up environment
- Set up a Python 3.12 environment. Python 3.12[Miniconda Distribution](https://www.anaconda.com/download)
- Install the required packages by using pip and the provided requirements.txt file
- Install Pytorch with Cuda 13.0 using the following command:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
- In case, there are problems with installing or running Pytorch then check their [instructions](https://pytorch.org/get-started/locally/).

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