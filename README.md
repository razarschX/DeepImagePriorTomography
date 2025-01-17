# Radon Transform and Deep Image Prior for Image Reconstruction

This repository contains MATLAB and Python implementations of the Radon Transform and Image Reconstruction techniques. The Python implementation further incorporates **Deep Image Prior (DIP)** to denoise corrupted sinograms and improve image reconstruction quality.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [MATLAB](#matlab-usage)
  - [Python](#python-usage)
- [Deep Image Prior](#deep-image-prior)
- [Results](#results)
- [References](#references)

## Introduction

The Radon transform is a powerful tool in medical imaging (e.g., X-ray tomography) for converting spatial domain data into a sinogram, representing line integrals through the image at various angles. The inverse Radon transform allows reconstructing the image from these projections.

In the Python implementation, we use **Deep Image Prior (DIP)**, a CNN-based technique, to improve the reconstruction quality from corrupted sinograms by leveraging the inherent structure of convolutional networks.

## Features

- Perform **Radon Transform** and **Inverse Radon Transform** to reconstruct images.
- Handle **corrupted sinograms** and recover clean images using **Deep Image Prior (DIP)**.
- Visualize original images, sinograms, and reconstructed images.
- Calculate **Mean Squared Error (MSE)** between original and reconstructed images.

## Dependencies

### MATLAB

Ensure you have MATLAB installed with the Image Processing Toolbox. The script is designed to run on any version that supports basic image processing and Radon Transform functions.

### Python

The Python implementation uses the following dependencies:

- `numpy`
- `matplotlib`
- `scikit-image`
- `torch` (PyTorch)
- `torchvision` (optional for future enhancements)

You can install the necessary Python libraries using:

```bash
pip install numpy matplotlib scikit-image torch torchvision
```

## Usage

### MATLAB Usage

1. Clone this repository.
2. Navigate to the MATLAB folder.
3. Run the script in MATLAB:

```matlab
main()
```

4. You will be prompted to enter the image file path. Make sure the image is grayscale or can be converted to grayscale.

### Python Usage

1. Clone this repository.
2. Navigate to the Python folder.
3. Run the script in a Python environment:

```bash
python Xray_tomography Inverse Problem.py
```

4. You will be prompted to enter the path to the corrupted image file. The program will perform the Radon Transform, apply noise to the sinogram, and use DIP to reconstruct the image.

## Deep Image Prior

The Deep Image Prior (DIP) is a neural network that implicitly regularizes the reconstruction by leveraging its architecture. Unlike traditional supervised learning models, DIP doesn't require any pre-training and is solely optimized to recover a clean image from the corrupted data.

**Steps:**
1. Create a simple U-Net-like CNN architecture.
2. Train the network using the corrupted sinogram as input and the ground-truth clean sinogram as a target.
3. After training, reconstruct the image using the denoised sinogram.

### Network Architecture

- A simple U-Net-like architecture consisting of convolutional and transposed convolutional layers.
- Loss function: **Mean Squared Error (MSE)**.
- Optimizer: **Adam** with a learning rate of 0.01.

## Results

Here are the key results visualized in both the MATLAB and Python implementations:

- **Original Image**: The input image provided by the user.
- **Sinogram (Radon Transform)**: The result of applying the Radon Transform to the image.
- **Noisy Sinogram**: The sinogram with added Gaussian noise.
- **Denoised Sinogram** (Python only): The output of the Deep Image Prior model.
- **Reconstructed Image**: The result of the inverse Radon Transform applied to the clean or denoised sinogram.

You can compare the reconstructed image to the original image and assess the quality using the MSE metric.
