
# MASNet: An enhanced approach for detecting small targets in complex imagery

This repository contains the source code for the paper "MASNet: An enhanced approach for detecting small targets in complex imagery", published in The Visual Computer journal.

## Installation

**To install YOLOv9, run the following commands:**
```bash
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9
pip install -r requirements.txt
```
##  primary contributions

1. **SPDConv Module**: 
   To better capture crucial structural information from input data, we employ SPDConv to substitute the original convolutional module. This substitution effectively enhances the model's feature extraction capabilities in complex environments and improves its capacity to detect faint and small targets.

2. **MLCA Module**:
   The MLCA module is incorporated into the backbone architecture, allowing the model to effectively prioritize different channels within the feature map at both local and global levels. This enhancement enables the model to capture more intricate and comprehensive feature information, thereby bolstering its capacity for detailed processing.

3. **MSDA Module**:
   MSDA is incorporated into the model's Head component to enhance its capacity for multi-scale feature learning, effectively processing target information across different scales, and improving the model's robustness in detecting small targets within complex scenes.

## Datasets

The datasets used in this paper are VOC2007, VOC2012, and VisDrone2019. They can be downloaded from the following links:
- [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007)
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012)
- [VisDrone2019](https://github.com/VisDrone/VisDroneDataset)

## Training

To train the model, please download the datasets mentioned above and use [YOLO2VOC](https://github.com/jahongir7174/YOLO2VOC) to convert the XML labels in the datasets to TXT format, making them suitable for YOLO training. Then, create a `data` folder to configure the training and validation sets.

## Implementation Details

The research was conducted within the Ubuntu 20.04 operating system environment. Utilizing the PyTorch 1.11.0 neural network framework, the experimental setup was hardware-supported by a GPU RTX 3090 24G, running CUDA 11.3, within a Python 3.8 computational context. The training protocol was executed with a batch size of 16 and a 300-epoch duration.
