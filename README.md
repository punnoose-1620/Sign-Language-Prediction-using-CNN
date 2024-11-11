# Sign Language Prediction using CNN

This project implements an image classification model using a Convolutional Neural Network (CNN) in TensorFlow/Keras. The model is designed to predict categorical labels from images processed via the ImageDataGenerator utility in Keras.

1. Contents
2. Overview
3. Model Architecture
4. Training Process
5. Evaluation Results
6. How to Use
7. Dependencies

## Overview

The Prediction Model is a deep learning-based image classifier developed in Python. It leverages a CNN for feature extraction and classification on an image dataset provided in a specific directory structure. The model's performance is assessed using accuracy and loss metrics on both training and validation datasets.

## Model Architecture

The architecture follows a standard CNN structure:

- Convolutional Layers: Stacks of convolutional layers to extract spatial features from the input images.
- Batch Normalization: Normalizes the output of convolutional layers for faster convergence.
- Pooling Layers: MaxPooling layers reduce the spatial dimensionality, retaining only the most significant features.
- Dropout: Dropout layers are included to prevent overfitting.
- Fully Connected Layers: Dense layers at the end for classification based on extracted features.

Key components include:

- Conv2D Layers: For feature extraction.
- Batch Normalization: For stable training.
- MaxPooling2D: For downsampling.
- Dropout Layers: To mitigate overfitting.
- Dense Layer: Final layer with softmax activation for outputting class probabilities.

## Training Process

The model is compiled and trained using the following setup:

- Loss Function: Categorical crossentropy, suitable for multi-class classification.
- Optimizer: Adam optimizer for adaptive learning.
- Callbacks: Learning rate reduction and early stopping based on validation performance.

The model is trained for 10 epochs with real-time data augmentation provided by ImageDataGenerator, which applies transformations such as rotation, zoom, and flip to enhance the model's generalization ability.

## Evaluation Results

The model achieved the following metrics on the validation set:

Epoch	Loss	Accuracy
1	0.68	90.30%
2	0.12	96.00%
3	0.10	96.33%
...	...	...
10	0.08	96.33%

Final evaluation on a separate test set yielded:

Loss: 0.34
Accuracy: 80%

## How to Use

Clone the repository:

  ```bash
  git clone <repository-url>
  cd <repository-name>
```

Install dependencies:

  ```bash
  pip install -r requirements.txt
```

Prepare the dataset in the required format with separate folders for each category.

Run the notebook or script to start training and evaluating the model.

The trained model will be saved as best_model.h5.

## Dependencies:

- TensorFlow
- Keras
- numpy
- ImageDataGenerator
