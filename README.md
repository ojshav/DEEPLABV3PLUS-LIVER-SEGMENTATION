# Liver Tumor Segmentation using Deep Learning

This repository contains code for liver tumor segmentation using deep learning techniques. The segmentation model is implemented in PyTorch with the DeepLabV3Plus architecture.

## Overview

The main file in this repository is `deeplabv3plus.ipynb`, which contains the entire implementation of the liver tumor segmentation model. This notebook includes the dataset loading, model definition, training loop, evaluation, and inference functions.

## Code Explanation

### Imports

The code begins with importing necessary libraries:

- `torch`: PyTorch library for deep learning tasks.
- `numpy`: NumPy library for numerical computations.
- `albumentations`: Albumentations library for image augmentation.
- `nibabel`: NiBabel library for reading and writing NIfTI files.
- `matplotlib.pyplot`: Matplotlib library for visualization.
- `segmentation_models_pytorch`: Library containing pre-defined segmentation models.

These libraries are essential for various tasks such as data loading, model training, and visualization.

### Dataset Class

The `CustomSegmentationDataset` class is responsible for loading and preprocessing the liver tumor segmentation dataset. It performs the following tasks:

- Loads CT images and corresponding tumor masks from NIfTI files.
- Applies transformations such as resizing and normalization to the images and masks.
- Preprocesses the images and masks for training and evaluation.

### Model Definition

The liver tumor segmentation model is defined using the DeepLabV3Plus architecture, implemented with the `smp.DeepLabV3Plus` class from the `segmentation_models_pytorch` library. This class initializes the model with the specified number of classes (tumor and background) and input channels.

### Training Loop

The `train` function implements the training loop. It iterates over the training data, computes the loss using the Cross Entropy Loss, performs backpropagation, and updates the model parameters using the Adam optimizer. The training process is repeated for multiple epochs to optimize the model parameters.

### Evaluation

The code includes evaluation metrics such as Intersection over Union (IoU) and pixel accuracy to assess the performance of the model on the validation set. These metrics are calculated using the `Metrics` class, which compares the predicted masks with the ground truth masks.

### Inference

After training, the trained model can be used for inference on new CT images to generate tumor segmentation masks. The `inference` function takes a DataLoader containing test images as input and returns the predicted masks using the trained model.



