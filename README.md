# Liver Tumor Segmentation using Deep Learning

This repository contains code for liver tumor segmentation using deep learning techniques. The segmentation model is implemented in PyTorch with the DeepLabV3Plus architecture.

## Overview

The main file in this repository is `deeplabv3plus.ipynb`, which contains the entire implementation of the liver tumor segmentation model. This file includes the dataset loading, model definition, training loop, evaluation, and inference functions.

## Code Explanation

### Imports

The code begins with importing necessary libraries including PyTorch, NumPy, Albumentations, Nibabel, Matplotlib, and others. These libraries are used for various tasks such as data loading, model training, and visualization.

### Dataset Class

The `CustomSegmentationDataset` class defined in the code is responsible for loading and preprocessing the liver tumor segmentation dataset. It reads the CT images and corresponding tumor masks from NIfTI files and applies transformations as required.

### Model Definition

The liver tumor segmentation model is defined using the DeepLabV3Plus architecture, which is implemented using the `smp.DeepLabV3Plus` class from the `segmentation_models_pytorch` library. This class initializes the model with the specified number of classes and input channels.

### Training Loop

The `train` function in the code implements the training loop. It iterates over the training data, computes the loss, performs backpropagation, and updates the model parameters using the Adam optimizer. The training process is repeated for multiple epochs.

### Evaluation

The code includes evaluation metrics such as Intersection over Union (IoU) and pixel accuracy to assess the performance of the model on the validation set. These metrics are calculated using the `Metrics` class.

### Inference

After training, the trained model can be used for inference on new CT images to generate tumor segmentation masks. The `inference` function in the code takes a DataLoader containing test images as input and returns the predicted masks.
