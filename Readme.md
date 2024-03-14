# Binary Classification

## Image Binary Classification Project
This project focuses on image classification using TensorFlow. It specifically addresses binary classification tasks where images are classified into one of two categories. In this case, the project is tailored for distinguishing between two classes of persons.

## Overview
The project employs a convolutional neural network (CNN) architecture for image classification. TensorFlow, a popular deep learning framework, is utilized for implementing the CNN model. The model architecture includes convolutional layers, batch normalization, ReLU activation, max pooling, and fully connected layers.

## Model Architecture
The CNN model architecture consists of:

Three convolutional layers with increasing depth for feature extraction.
Batch normalization layers for normalizing the activations.
ReLU activation functions to introduce non-linearity.
Max pooling layers for downsampling and extracting dominant features.
Fully connected layers for classification.
Binary classification output layer using a sigmoid activation function.
Data Preparation
The dataset is organized into two classes, each containing images of persons. Data augmentation techniques such as image sharpening and flipping are applied to increase the diversity of the training dataset.

## Training
The model is trained using the Adam optimizer with a binary cross-entropy loss function. Training is performed over multiple epochs with a specified batch size. Additionally, a subset of the training data is used for validation during the training process.

## Evaluation
After training, the model is evaluated using a separate test dataset. Evaluation metrics including F1 score and accuracy are computed to assess the performance of the model on unseen data.

## Usage
To use this project:

Ensure TensorFlow and other required dependencies are installed.
Prepare the dataset following the specified structure.
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

## Contact:

For any inquiries, please contact Nima87760@gmail.com.

