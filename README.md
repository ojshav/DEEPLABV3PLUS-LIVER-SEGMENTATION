
# Liver Tumor Segmentation using Deep Learning

This repository contains code for liver tumor segmentation using deep learning techniques. The segmentation model is implemented in PyTorch with various architectures including DeepLabV3Plus, EfficientNet, ResNet50, ResNet34, and MobileNet.

## Overview

The main files in this repository are Jupyter notebooks named after each architecture, such as `deeplabv3plus.ipynb`, `efficientnet.ipynb`, `resnet50.ipynb`, `resnet34.ipynb`, and `mobilenet.ipynb`. Each notebook contains the implementation of the liver tumor segmentation model with the respective architecture. These notebooks include the dataset loading, model definition, training loop, evaluation, and inference functions.

## Code Explanation

### Imports

The code begins with importing necessary libraries:

- `torch`, `numpy`, `albumentations`, `nibabel`, `matplotlib.pyplot`: Essential libraries for deep learning tasks, numerical computations, image augmentation, file I/O, and visualization.
- `segmentation_models_pytorch`: Library containing pre-defined segmentation models.

### Dataset Class

Each notebook contains a `CustomSegmentationDataset` class responsible for loading and preprocessing the liver tumor segmentation dataset. It performs tasks such as loading CT images and corresponding tumor masks from NIfTI files, applying transformations, and preprocessing the data for training and evaluation.

### Model Definition

The liver tumor segmentation models are defined using different architectures such as DeepLabV3Plus, EfficientNet, ResNet50, ResNet34, and MobileNet. These architectures are implemented with the respective classes from the `segmentation_models_pytorch` library. Each notebook initializes the model with the specified number of classes (tumor and background) and input channels.

### Training Loop

The `train` function implements the training loop. It iterates over the training data, computes the loss using the Cross Entropy Loss, performs backpropagation, and updates the model parameters using the Adam optimizer. The training process is repeated for multiple epochs to optimize the model parameters.

### Evaluation

Each notebook includes evaluation metrics such as Intersection over Union (IoU) and pixel accuracy to assess the performance of the model on the validation set. These metrics are calculated using the `Metrics` class, which compares the predicted masks with the ground truth masks.

### Inference

After training, the trained models can be used for inference on new CT images to generate tumor segmentation masks. The `inference` function takes a DataLoader containing test images as input and returns the predicted masks using the trained model.

## Results

The following table summarizes the results of each model on the liver tumor segmentation task:

| Model                     | Train Time (mins) | Train Loss | Train PA | Train IoU | Validation Loss | Validation PA | Validation IoU | Test Loss | Test PA | Test IoU |
|----------------------------|-------------------|------------|----------|-----------|-----------------|---------------|----------------|-----------|---------|----------|
| DeepLabV3Plus             | 1.713             | 0.010      | 0.932    | 0.967     | 0.011           | 0.930         | 0.967          | 0.010     | 0.996   | 0.941    |
| DeepLabV3Plus + EfficientNet | 2.866             | 0.012      | 0.933    | 0.956     | 0.014           | 0.926         | 0.957          | 0.013     | 0.995   | 0.919    |
| DeepLabV3Plus + ResNet50  | 5.135             | 0.010      | 0.932    | 0.970     | 0.011           | 0.935         | 0.968          | 0.008     | 0.997   | 0.927    |
| DeepLabV3Plus + ResNet34  | 1.972             | 0.010      | 0.932    | 0.968     | 0.011           | 0.933         | 0.966          | 0.010     | 0.996   | 0.928    |
| DeepLabV3Plus + MobileNet | 1.848             | 0.012      | 0.933    | 0.957     | 0.018           | 0.927         | 0.946          | 0.016     | 0.994   | 0.874    |

```
