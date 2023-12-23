# Emotion Recognition with DenseNet169

This repository contains code for training a deep-learning model for emotion recognition using DenseNet169. The model is trained on a dataset of facial images representing different emotions. The code includes data loading, preprocessing, model training, fine-tuning, and evaluation.

## Table of Contents

- [Hyperparameters and Directories](#hp)
- [Data Loading and Pre-processing](#data)
- [DenseNet169 Transfer Learning](#model)
- [Training and Fine-Tuning](#train)
- [Visualizing Results](#vis)

## Hyperparameters and Directories

The hyperparameters and directories used in the code are specified in this section. It includes parameters such as image dimensions, batch size, learning rate, and more.

## Data Loading and Pre-processing

This section involves loading and pre-processing the image data. The code uses TensorFlow and Keras for image data augmentation and normalization.

## DenseNet169 Transfer Learning

The model architecture is based on DenseNet169, a pre-trained model on ImageNet. The code defines the model architecture, compiles it, and freezes the feature extraction layers for the initial training.

## Training and Fine-Tuning

The model is trained with the frozen layers, and then fine-tuning is performed by unfreezing some layers. Training plots, including accuracy and loss, are visualized in this section.

## Visualizing Results

This section includes the evaluation of the trained model, including a confusion matrix, classification report, and a multiclass AUC curve.

## Usage

To use this code, follow these steps:

1. Clone the repository.
2. Install the required libraries specified in the code.
3. Set the appropriate directories for training and testing data.
4. Run the code sections sequentially.

## Results

The model achieves good performance, as demonstrated by the evaluation metrics and visualizations.

## Acknowledgments

Thanks to Sanskar Hasija for the Dataset. Link to the dataset: https://www.kaggle.com/code/odins0n/emotion-detection

