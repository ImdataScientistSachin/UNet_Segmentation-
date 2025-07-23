
# About Me
Hi, I'm Sachin 👋 — a Data Scientist passionate about harnessing deep learning and AI for impactful solutions in medical imaging and computer vision.

# 👩‍💻 Currently working on:
Developing and optimizing U-Net segmentation models to accurately detect and track skin lesions. Focused on applying advanced deep learning architectures for automated lesion delineation and supporting longitudinal analysis using the HAM10000 dataset.

# 🧠 Currently learning:
MLOps strategies for seamless deployment, monitoring, and scaling of machine learning pipelines in real-world environments.

# 👯‍♀️ Open to collaborations on:
Innovative medical imaging analysis, robust semantic segmentation projects, and workflows for healthcare AI.

# 🤔 Looking for help with:
Exploring Generative AI (GENAI), particularly for creating synthetic medical imagery and utilizing large language models for enhanced data annotation.

# 💬 Ask me about:
AI, Deep Learning architectures, Computer Vision, and real-world applications of data science in healthcare.

# 📫 How to reach me:
[imdatascientistsachin@gmail.com] | [LinkedIn Profile]


## 😄 Pronouns:
He / Him

# ⚡️ Fun fact:
I enjoy creating AI-art and experimenting with neural style transfer in my free time.

# Project Title
skin lesion tracking using U-Net Segmentation

Overview
This repository presents a robust deep learning pipeline for the segmentation and longitudinal tracking of skin lesions using a custom U-Net architecture, carefully trained and validated on the industry-standard HAM10000 dataset. The goal is to automate the pixel-precise identification and monitoring of moles, melanomas, and other skin abnormalities—supporting the early detection of serious conditions, improving diagnostic consistency, and enabling large-scale research.




## Problem Statement
Manual outlining and measurement of skin lesions is subjective, laborious, and susceptible to variability. This project automates the process, providing:

Accurate boundary detection, even with diverse lesion presentations.

Consistent area and color extraction to aid in patient monitoring over time.

Automated alerts for significant changes, supporting clinical workflows.

##  Dataset Reference
HAM10000 ("Human Against Machine with 10,000 training images"):
          A benchmark dataset containing 10,000 dermoscopic images of pigmented skin lesions with expert annotations, supporting research in melanoma detection and segmentation.

# Model Architecture
U-Net based on the 2015 paper by Ronneberger et al.:
A symmetric encoder-decoder convolutional network with skip connections to preserve spatial details critical for biomedical image segmentation.

Input images resized to 128×128 or 256×256 pixels.

Binary output masks separating lesion foreground from background.

## Features & Pipeline
Data preprocessing: Image resizing, normalization, and mask binarization.

Data augmentation: Random horizontal/vertical flips, brightness/contrast adjustments using Albumentations.

Generator-based training: Efficient batch loading and real-time augmentation.

Model training: Adam optimizer with binary cross-entropy loss; early stopping and model checkpointing.

Inference: Predict binary masks on new images.

Feature extraction: Compute lesion area (in pixels) and mean RGB color inside lesion.

Tracking: Patient-level lesion feature tracking over time using pandas DataFrames.

Alerts: Configurable threshold alerts for clinically relevant lesion changes (default ±15% area change).

Visualization: Plot lesion growth and percent area change across visits.

# Quick Start
  ### Requirements
    Python 3.7+
    TensorFlow 2.x and Keras
    NumPy, pandas, OpenCV, scikit-image, albumentations, matplotlib

## Installation
git clone https://github.com/ImdataScientistSachin/UNet_Segmentation.git
cd skin-lesion-unet-tracking
pip install -r requirements.txt

## Data Preparation
Download HAM10000 dataset and organize images and masks as follows:

text
data/raw/HAM10000_images_part_1/
data/raw/HAM10000_images_part_2/
data/raw/masks/
data/processed/images/
data/processed/masks/


# Run provided preprocessing scripts to resize and pair images and masks.

### Training Example

  from skin_lesion_tracking_using_UNet_segmentation import unet, DataGen
model = unet(input_size=(128,128,3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_gen = DataGen(train_imgs, train_masks, batch_size=16, augment=True)
val_gen = DataGen(val_imgs, val_masks, batch_size=16, augment=False)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[
        keras.callbacks.ModelCheckpoint('unet_best.h5', save_best_only=True),
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ],
    verbose=1
)

    

# Inference & Tracking
Predict lesions on new images.

Extract lesion area and mean color.

Track longitudinal changes; generate alerts if area shifts exceed set thresholds.

# Visualization
Overlay segmentation masks on images for ground truth comparison.

Time-series plots reveal lesion area or color trends over multiple visits.

# Applications
Dermatology research and annotation at scale

Automated and objective lesion follow-up

Telemedicine and remote patient monitoring

AI-assisted clinical decision support

# Citation
If you use this repository or its methods, please cite:

HAM10000 dataset:
Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 Dataset: A Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions. Scientific Data, 5, 180161.

U-Net architecture:
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In MICCAI.

