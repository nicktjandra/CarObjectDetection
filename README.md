# Tools for Car Object Detection
Flatiron Data Science Capstone Project

<img width="1109" alt="Screen Shot 2023-10-03 at 12 13 27 PM" src="https://github.com/nickthetj/CarObjectDetection/assets/37059423/bf1b8802-b453-44ea-8dde-fe1a0ddcfe1d">


## Overview
Object recognition is the linchpin for self-driving cars and traffic safety cameras. It enables self-driving cars to identify and respond to objects like pedestrians and vehicles in real-time, ensuring safer autonomous driving. Additionally, it helps traffic safety cameras monitor and enforce traffic rules, enhancing road safety and efficiency. This notebook will walk through three object detection models that can identify and draw bounding-boxes around cars on the road: YOLOv8, R-CNN, and Bounding Box Regression (BB). The YOLOv8 model was implemented with the pre-packaged model from Ultralytics, and R-CNN and BB were built from the ground up. We will walk through differences between each model on a conceptual level, their implementation and respective evaluation scores. 

## The Data
The data was taken from a kaggle dataset consisting of roughly a thousand street view images divided into test and training sets. Some of these images consisted of cars and others did not. The dataset also included bounding box informations associated with each image. There were no labels for this dataset as there was only one class of object, a car. 

To download this dataset visit the following link: https://www.kaggle.com/datasets/sshikamaru/car-object-detection

## Data Preparation
In the notebook, the following preprocessing steps have been applied to each of the models:

For YOLOv8:
The input images were resized or padded to a fixed size (e.g., 416x416 pixels).
Pixel values were normalized for consistent scaling.
The bounding box information from the dataset was reorganized into yaml files for the yolo algorithm to process.

Regarding R-CNN:
Region proposals were generated, employing methods like selective search.
Each proposal was cropped and resized to match the model's input size.
Pixel values were normalized to maintain data uniformity.

Concerning Bounding Box Regression:
It was assumed that objects were already localized within the input images.
Optional adjustments to bounding box coordinates were made when necessary.
Bounding box coordinates or relevant features were normalized as required.
These preprocessing steps ensure that the data is suitably prepared for each model within the notebook.

## Modeling
In this notebook, there are three models for car detection: YOLOv8, R-CNN, and Bounding Box Regression.

YOLOv8 (You Only Look Once):
YOLOv8 is an efficient real-time object detection model. It divides the input image into a grid and predicts bounding boxes and class probabilities for objects within each grid cell. YOLOv8 uses a single neural network to simultaneously predict bounding boxes and class labels, making it fast and accurate. It operates by regressing bounding box coordinates and class probabilities directly from the image features.

R-CNN (Region-based Convolutional Neural Network):
R-CNN is a two-step object detection approach. It first generates region proposals from the image using a selective search or similar method. Then, each region proposal is individually passed through a CNN to extract features. These features are then used to classify and refine the bounding box of the detected object. R-CNNs are accurate but computationally expensive due to the two-stage process.

Bounding Box Regression:
Bounding Box Regression is a simple technique that involves training a model to predict adjustments to the coordinates of bounding boxes. Initially, a bounding box is defined around an object's region. The model then learns to predict corrections to the box's coordinates, refining its position and size. This model however is limited because it can only predict one bounding box in an image that should contain multiple vehicles. 
