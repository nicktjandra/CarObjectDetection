# Tools for Car Object Detection

<img width="1109" alt="Screen Shot 2023-10-03 at 12 13 27 PM" src="https://github.com/nickthetj/CarObjectDetection/assets/37059423/bf1b8802-b453-44ea-8dde-fe1a0ddcfe1d">
(Source: Zapp2Photo/Shutterstock.com)

## Overview
Object recognition is the linchpin for self-driving cars. It enables self-driving cars to identify and respond to objects like pedestrians and vehicles in real-time, ensuring safer autonomous driving. This notebook will walk through three object detection models that can identify and draw bounding-boxes around cars on the road: YOLOv8, R-CNN, and Bounding Box Regression (BB). The YOLOv8 model was implemented with the pre-packaged model from Ultralytics, and R-CNN and BB were built from the ground up. We will walk through differences between each model on a conceptual level, their implementation and respective evaluation scores. 

## Business Problem
These object detection models were developed as a tool for autonomous vehicles and traffic safety cameras to enable the cameras to perceive and understand its environment by identifying and locating objects of interest in real-time. Specifically in identifying and locating where cars are in a picture.

## The Data
The data was taken from a kaggle dataset consisting of 1176 street view images divided into test and training sets. Some of these images consisted of cars and others did not. The dataset also included bounding box informations associated with each image. There were no labels for this dataset as there was only one class of object, a car. 

To download this dataset visit the following link: https://www.kaggle.com/datasets/sshikamaru/car-object-detection

## Data Preparation
In the notebook, different preprocessing steps have been applied for each of the models:

For YOLOv8:
The input images were resized or padded to a fixed size (e.g., 416x416 pixels).
Pixel values were normalized for consistent scaling.
The bounding box information from the dataset was reorganized into yaml files for the yolo algorithm to process.

Regarding R-CNN:
Region proposals were generated, employing methods like selective search.
Each proposal was cropped and resized to match the model's input size.
Pixel values were normalized to maintain data uniformity.

Concerning Bounding Box Regression:
There was not much data preprocessing needed for Bounding Box regression. For the bounding boxes I extracted them from the csv file, ensuring that non of the boxes were outside of the image dimensions. And for the images, I loaded the each image with 224x224 dimensions then normalized the color values per pixel to between 0 and 1.

## Modeling
In this notebook, there are three models for car detection: YOLOv8, R-CNN, and Bounding Box Regression.

YOLOv8 (You Only Look Once):
YOLOv8 is an efficient real-time object detection model. It divides the input image into a grid and predicts bounding boxes and class probabilities for objects within each grid cell. YOLOv8 uses a single neural network to simultaneously predict bounding boxes and class labels, making it fast and accurate. It operates by regressing bounding box coordinates and class probabilities directly from the image features.

R-CNN (Region-based Convolutional Neural Network):
R-CNN is a two-step object detection approach. It first generates region proposals from the image using a selective search or similar method. Then, each region proposal is individually passed through a CNN to extract features. These features are then used to classify and refine the bounding box of the detected object. R-CNNs are accurate but computationally expensive due to the two-stage process.

Bounding Box Regression:
Bounding Box Regression is a simple technique that involves training a model to predict adjustments to the coordinates of bounding boxes. Initially, a bounding box is defined around an object's region. The model then learns to predict corrections to the box's coordinates, refining its position and size. This model however is limited because it can only predict one bounding box in an image that should contain multiple vehicles. 


## Evaluation

## Conclusions

I would recommend the YOLOv8 over R-CNN or Bounding Box model for single-class object detection due to the speed, simplicity, and efficiency in handling objects of varying sizes and aspect ratios. These models are single-shot detectors, enabling real-time processing, and require less data for training, making them practical for limited datasets. However, the choice should be based on your specific requirements, as RCNN-based models excel in precise localization and larger datasets.

## Next Steps
Given more time, these object detection models could be expanded to include a wider range of objects like pedestrians, street signs, and various types of vehicles. This enhancement would require much more data and training time but would make self-driving systems more aware and capable, improving road safety and adherence to traffic rules.

Tangential to improving the model, we could also develop a system with the ability to measure the precise distance between the host vehicle and detected cars and calculate the velocities of objects within the camera's field of view. Additionally, the capability to discern whether a detected car is moving away from or approaching the host vehicle would be very helpful, moving towards a more sophisticated autonomous driving system. 

## Repository Structure
```
├── data
├── README.md
├── BoundingBox_Final.ipynb
├── YOLOv8_Final.ipynb
├── RCNN_Final.ipynb
├── .gitignore
├── LICENSE
└── presentation.pdf
```
