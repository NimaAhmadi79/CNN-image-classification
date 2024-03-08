# Ten-Class Image Classification Project
This project extends the previous binary classification task to classify images into ten different categories using a convolutional neural network (CNN) implemented in TensorFlow.

## Overview
The project aims to classify images into one of ten classes using a CNN architecture. TensorFlow is utilized for building and training the model.

## Model Architecture
The CNN model architecture comprises convolutional layers, batch normalization, ReLU activation, max pooling, and fully connected layers.

## Data Preparation
The dataset consists of images categorized into ten classes. Data augmentation techniques such as image sharpening and flipping are applied to increase the diversity of the training dataset.

## Training
Training is performed using the Adam optimizer with categorical cross-entropy loss. The model is trained over multiple epochs with a specified batch size.

## Evaluation
Evaluation metrics including loss, accuracy, and F1 score are computed to assess the performance of the model on a separate test dataset.

## Usage
To use this project:

Install TensorFlow and required dependencies.
Prepare the dataset according to the specified structure.
Adjust hyperparameters and model architecture if necessary.
Train the model using the provided training script.
Evaluate the trained model on the test dataset.
Fine-tune the model or experiment with different configurations as needed.

## Requirements
TensorFlow
NumPy
OpenCV
Matplotlib
scikit-learn

## Author
Nima Ahmadi

