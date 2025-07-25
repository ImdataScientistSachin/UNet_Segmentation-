# -*- coding: utf-8 -*-
"""skin lesion tracking using U-Net Segmentation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KhSy1lxh-z3iU4cCJsWlWtFizXo8kkD3

# skin lesion tracking using U-Net Segmentation
"""

# skin lesion tracking using U-Net Segmentation

"""
    Problem Addressed:
Early detection and consistent monitoring of skin lesions (moles, wounds)
 are critical for preventing serious conditions such as melanoma.
 Manual tracking is inconsistent and can be a barrier to seeking timely medical advice.
"""

# Mount Google Drive for faseter training

from google.colab import drive
drive.mount('/content/drive')

# check Gpu availability

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# check the dataset present or not

import os

# Example path (adjust as needed)
data_dir = "/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/raw/HAM10000"
print(os.listdir(data_dir))



# Install segmentation libraries & Import Libraries
!pip install albumentations

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import train_test_split

raw_dir = "/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/raw/HAM10000"
img_dirs = [
    os.path.join(raw_dir, "HAM10000_images_part_1"),
    os.path.join(raw_dir, "HAM10000_images_part_2"),
]

image_files = []
for directory in img_dirs:
    for f in os.listdir(directory):
        if f.lower().endswith('.jpg'):  # or '.png' if using converted images
            image_files.append(os.path.join(directory, f))

print(f"Total images found: {len(image_files)}")

# Check main project path
print(os.listdir('/content/drive/MyDrive/'))

# Check your project folder and data structure
project_folder = 'skin lesion tracking using U-Net Segmentation-project'
project_path = f'/content/drive/MyDrive/{project_folder}'
print(os.listdir(project_path))

# Check subfolders
processed_path = os.path.join(project_path, 'data/processed')
print(os.listdir(processed_path))



"""# use when you have a better gpu performance

# preprocessing the images and segmentation masks

# constant dir paths

RAW_IMG_DIR_1 = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/raw/HAM10000/HAM10000_images_part_1/'
RAW_IMG_DIR_2 = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/raw/HAM10000/HAM10000_images_part_2/'
RAW_MASK_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/raw/masks/' # If masks provided as images
PROC_IMG_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/processed/images/'
PROC_MASK_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/processed/masks/'
IMG_SIZE = 128  # or 256, depending on model's expected input

os.makedirs(PROC_IMG_DIR, exist_ok=True)
os.makedirs(PROC_MASK_DIR, exist_ok=True)



# Combine both image parts
raw_image_folders = [RAW_IMG_DIR_1, RAW_IMG_DIR_2]
all_images = []
for folder in raw_image_folders:
    all_images.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')])

# Process images
for img_path in tqdm(all_images, desc="Processing Images"):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    fname = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(PROC_IMG_DIR, fname + '.png')  # standardize to .png
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Process masks if provided
# You may need to adapt this part if mask filenames differ or need to be generated.
for img_path in tqdm(all_images, desc="Processing Masks"):
    fname = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(RAW_MASK_DIR, fname + '_segmentation.png')
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        # Binarize if necessary
        mask = (mask > 128).astype('uint8') * 255
        out_path = os.path.join(PROC_MASK_DIR, fname + '.png')
        cv2.imwrite(out_path, mask)
    else:
        print(f"Mask missing for {fname}, skipping.")

RAW_IMG_DIR_2 = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/raw/HAM10000/HAM10000_images_part_2/'
RAW_MASK_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/raw/HAM10000/masks/' # If masks provided as images
PROC_IMG_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/processed/images/'
PROC_MASK_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/processed/masks/'
IMG_SIZE = 128  # or 256, depending on your model's expected input

os.makedirs(PROC_IMG_DIR, exist_ok=True)
os.makedirs(PROC_MASK_DIR, exist_ok=True)



# Combine both image parts
raw_image_folders = [RAW_IMG_DIR_1, RAW_IMG_DIR_2]
all_images = []
for folder in raw_image_folders:
    all_images.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')])

# Process images
for img_path in tqdm(all_images, desc="Processing Images"):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    fname = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(PROC_IMG_DIR, fname + '.png')  # standardize to .png
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Process masks if provided
#  need to adapt this part if mask filenames differ or need to be generated.
for img_path in tqdm(all_images, desc="Processing Masks"):
    fname = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(RAW_MASK_DIR, fname + '_segmentation.png')
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        # Binarize if necessary
        mask = (mask > 128).astype('uint8') * 255
        out_path = os.path.join(PROC_MASK_DIR, fname + '.png')
        cv2.imwrite(out_path, mask)
    else:
        print(f"Mask missing for {fname}, skipping.")

A mask is a grayscale (single-channel) image where:Each pixel represents a label for its corresponding pixel in the original input image.

For binary segmentation, pixels are either:
Foreground (lesion/object): typically assigned a value of 255 (white).
Background: assigned a value of 0 (black).
"""

# we use this code to copy our files to temp storage od google colab for FASTER TRAINNING

!cp -r "/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/processed/images/." /content/images/
!cp -r "/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/processed/masks/." /content/masks/

# Datasets path and data loading

IMG_SIZE = 128  # Use 256 if your U-Net expects that

# Paths (adjust as necessary)
DATA_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/raw/HAM10000/'
PROCESSED_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/processed/'
PROCESSED_IMG_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/processed/images/'
PROCESSED_MASK_DIR = '/content/drive/MyDrive/skin lesion tracking using U-Net Segmentation-project/data/processed/masks/'



# Example: we  downloaded from Kaggle, this CSV contains image list and metadata
csv_path = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
df = pd.read_csv(csv_path)

# Example: images are .jpg and masks are .png in the processed directory
image_folder = os.path.join(PROCESSED_DIR, 'images')
mask_folder = os.path.join(PROCESSED_DIR, 'masks')

# Gather file paths (adapt as needed)
images = sorted([os.path.join(image_folder, fname) for fname in os.listdir(image_folder)])
masks  = sorted([os.path.join(mask_folder, fname) for fname in os.listdir(mask_folder)])
print(f"Found {len(images)} images and {len(masks)} masks")


# we Use local disk, not Google Drive, for fast training because our dataset is too large
PROC_IMG_DIR = '/content/images/'
PROC_MASK_DIR = '/content/masks/'

# Example: images are .jpg and masks are .png in the processed directory
image_folder_Temp = os.path.join(PROC_IMG_DIR, 'images')
mask_folder_Temp = os.path.join(PROC_MASK_DIR, 'masks')

# Gather file paths (adapt as needed)
images = sorted([os.path.join(image_folder_Temp, fname) for fname in os.listdir(image_folder)])
masks  = sorted([os.path.join(mask_folder_Temp, fname) for fname in os.listdir(mask_folder)])
print(f"Found {len(images)} images and {len(masks)} masks in Content ")

"""Pairing of Images and Masks"""

# === Robust Pairing of Images and Masks ===

"""
# in case we use our main stored dataaset

def extract_id(fname):
    Utility to extract the image ID from a filename (without extension).
    return os.path.splitext(os.path.basename(fname))[0]

# Gather all .png images and masks
image_files = [os.path.join(PROCESSED_IMG_DIR, f)
               for f in os.listdir(PROCESSED_IMG_DIR)
               if f.lower().endswith('.png')]
mask_files = [os.path.join(PROCESSED_MASK_DIR, f)
              for f in os.listdir(PROCESSED_MASK_DIR)
              if f.lower().endswith('.png')]

# Build dictionaries for quick lookup
image_dict = {extract_id(f): f for f in image_files}
mask_dict = {extract_id(f): f for f in mask_files}

# Only keep pairs where both image and mask exist
common_ids = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
paired_images = [image_dict[_id] for _id in common_ids]
paired_masks = [mask_dict[_id] for _id in common_ids]

print(f"Total paired samples: {len(paired_images)}")

"""


# if we use colab temp stored dataset
# Utility to extract the image ID from a filename (without extension)

def extract_id(fname):
    # Extract basename without extension
    return os.path.splitext(os.path.basename(fname))[0]

# Gather .png image and mask files

image_files = [os.path.join(PROC_IMG_DIR, f) for f in os.listdir(PROC_IMG_DIR) if f.lower().endswith('.png')]
mask_files  = [os.path.join(PROC_MASK_DIR, f) for f in os.listdir(PROC_MASK_DIR) if f.lower().endswith('.png')]

# Build pairing dictionaries
image_dict = {extract_id(f): f for f in image_files}
mask_dict  = {extract_id(f): f for f in mask_files}
common_ids = sorted(set(image_dict) & set(mask_dict))

paired_images = [image_dict[_id] for _id in common_ids]
paired_masks  = [mask_dict[_id] for _id in common_ids]

print(f"Paired dataset size: {len(paired_images)}")

# === Split into Train and Validation Sets ===
# spliting the dataset

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    paired_images, paired_masks, test_size=0.15, random_state=42
)

print(f"Train samples: {len(train_imgs)}, Validation samples: {len(val_imgs)}")

# load sample images for visualize


# Use the correct images directory (adjust as needed)
IMG_FOLDER = '/content/images/'
# List all .png image files
sample_imgs = [os.path.join(IMG_FOLDER, f) for f in os.listdir(IMG_FOLDER) if f.lower().endswith('.png')]

# we Choose how many images to show (e.g., 5)
num_samples = 5
selected_imgs = random.sample(sample_imgs, k=min(num_samples, len(sample_imgs)))

plt.figure(figsize=(15, 3))
for i, img_path in enumerate(selected_imgs):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Sample {i + 1}")
plt.suptitle('Random Sample Images')
plt.show()

"""### Apply  Data Preprocessing and Augmentation"""

# Apply  Data Preprocessing and Augmentation



# --- Step 4: Define data augmentation and preprocessing ---
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),   # Resize image & mask to fixed size for consistent input shape
    A.HorizontalFlip(p=0.5),        # Randomly flip horizontally to augment training
    A.VerticalFlip(p=0.5),          # Randomly flip vertically
    A.RandomBrightnessContrast(p=0.3),  # Randomly adjust brightness & contrast to help model generalize
])

def preprocess(img_path, mask_path, augment=True):
    """
    Load, optionally augment, normalize image and mask.
    """
    # Read image and convert from BGR(OpenCV default) to RGB (standard)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read mask as grayscale; expect pixel values 0 or 255
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise ValueError(f"Error reading image or mask: {img_path}, {mask_path}")

    # Apply augmentations if requested
    if augment:
        augmented = transform(image=img, mask=mask)
        img, mask = augmented['image'], augmented['mask']

    # Normalize image to range [0,1]
    img = img / 255.0

    # Binarize mask: lesion = 1, background = 0
    mask = (mask > 128).astype(np.float32)

    # Add channel dimension for mask (shape: H x W x 1)
    mask = np.expand_dims(mask, axis=-1)

    return img, mask

"""### Implement a data generator for efficient batch loading and augmentation

"""

# configure data generater

class DataGen(keras.utils.Sequence):
    """
    Keras data generator to yield batches of augmented images and masks on the fly.
    """
    def __init__(self, images, masks, batch_size=16, augment=True):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        # Pick batch slice
        batch_imgs = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.masks[idx * self.batch_size:(idx + 1) * self.batch_size]

        imgs, msks = [], []
        for img_path, mask_path in zip(batch_imgs, batch_masks):
            img, msk = preprocess(img_path, mask_path, augment=self.augment)
            imgs.append(img)
            msks.append(msk)

        return np.array(imgs), np.array(msks)

"""### Define the U-Net model architecture ---

"""

def unet(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Define U-Net model with encoder-decoder structure and skip connections.
    """
    inputs = keras.Input(input_size)

    # Encoder: Downsampling path
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, 3, activation='relu', padding='same')(c5)

    # Decoder: Upsampling path with skip connections
    u6 = layers.UpSampling2D(2)(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, 3, activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D(2)(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, 3, activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D(2)(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, 3, activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D(2)(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, 3, activation='relu', padding='same')(c9)

    # Output: Single channel binary mask with sigmoid activation
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)

    return keras.Model(inputs, outputs)

model = unet()
model.summary()

"""### Compile model with optimizer, loss, and metric"""

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Example DataGenerators usage

train_gen = DataGen(train_imgs, train_masks, batch_size=16, augment=True)
val_gen   = DataGen(val_imgs, val_masks, batch_size=16, augment=False)

# Setup callbacks for checkpointing best model & early stopping
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    '/content/unet_best.h5', save_best_only=True
)
earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Start training with train and val generators
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[checkpoint_cb, earlystopping_cb],
    verbose=1
)

# --- Define function to run segmentation on new images ---

def segment_image(model, img_rgb):
    """
    Predict segmentation mask for a given RGB image.
    Returns binary mask resized to IMG_SIZE.
    """
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)) / 255.0
    input_arr = np.expand_dims(resized, axis=0)
    pred_mask = model.predict(input_arr)[0, ..., 0]
    return (pred_mask > 0.5).astype(np.uint8)

# Extract lesion features (area + mean color) from mask ---

def extract_lesion_features(image_rgb, mask_bin):
    """
    Compute lesion area in pixels and average RGB color inside lesion mask.
    """
    labeled_mask = label(mask_bin)
    props = regionprops(labeled_mask)

    if not props:
        # No lesion detected
        return {'area_pixels': 0, 'mean_color': (0, 0, 0)}

    # Use the largest connected lesion area
    largest = max(props, key=lambda x: x.area)
    area = largest.area

    # Binary mask for largest lesion
    largest_mask = (labeled_mask == largest.label)

    lesion_pixels = image_rgb[largest_mask]

    mean_color = tuple(np.mean(lesion_pixels, axis=0).astype(int))

    return {'area_pixels': area, 'mean_color': mean_color}

# Track lesion's features over time and build DataFrame ---

def build_tracking_df(patient_id, records):
    """
    Given patient image records (date + RGB image),
    run segmentation and extract features for each date,
    returning a DataFrame with lesion growth info.
    """
    rows = []
    baseline_area = None
    for rec in sorted(records, key=lambda x: x['date']):
        mask = segment_image(model, rec['image_rgb'])
        feats = extract_lesion_features(rec['image_rgb'], mask)

        area = feats['area_pixels']
        mean_col = feats['mean_color']

        if baseline_area is None:
            baseline_area = area

        pct_change = ((area - baseline_area) / baseline_area) * 100 if baseline_area else 0

        rows.append({
            'patient_id': patient_id,
            'date': pd.to_datetime(rec['date']),
            'area_pixels': area,
            'mean_color_r': mean_col[0],
            'mean_color_g': mean_col[1],
            'mean_color_b': mean_col[2],
            'pct_area_change': pct_change
        })

    return pd.DataFrame(rows)

# Alert generation based on lesion changes ---
def alert_on_change(df, threshold=15.0):
    """
    Check if the lesion's area has changed beyond a threshold,
    signaling the need for dermatological consultation.
    """
    last_change = df.iloc[-1]['pct_area_change']

    if abs(last_change) >= threshold:
        return f"Alert! Lesion changed by {last_change:.1f}% since baseline. Please consult a dermatologist."
    else:
        return "No significant lesion changes detected."

# Visualization of lesion growth over time ---

def plot_growth(df):
    """
    Plot lesion area (pixels) and percent area change versus time.
    Creates dual y-axis plot for better visualization.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df['date'], df['area_pixels'], 'b-o', label='Lesion Area (pixels)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Area (pixels)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(df['date'], df['pct_area_change'], 'r--s', label='% Area Change')
    ax2.set_ylabel('% Area Change', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    plt.title(f"Lesion Growth Over Time - Patient {df['patient_id'].iloc[0]}")
    plt.grid(True)
    plt.show()

# Evaluate and Visualize Results

def plot_sample(img, mask_true, mask_pred):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img); plt.title('Image')
    plt.subplot(1,3,2); plt.imshow(mask_true.squeeze(), cmap='gray'); plt.title('Ground Truth')
    plt.subplot(1,3,3); plt.imshow(mask_pred.squeeze(), cmap='gray'); plt.title('Prediction')
    plt.show()

sample_img, sample_mask = preprocess(val_imgs[1], val_masks[0], augment=False)
pred_mask = model.predict(np.expand_dims(sample_img, 0))[0] > 0.5

plot_sample(sample_img, sample_mask, pred_mask)



"""
Problem Addressed by This Model


Clinical Background: Skin lesions—including moles, melanomas, wounds,
 and rashes—can change over time, and early detection of concerning features is vital for diagnosis,
  patient monitoring, and treatment planning in dermatology.
   Manual delineation (drawing borders) of lesions in clinical images is time-consuming, subjective,
   and reliant on clinical expertise. There is a growing need for automated, reliable, and accurate methods to segment (delineate) lesions in skin photographs for better tracking, screening, and analysis.

Technical Challenge : The primary problem this model addresses is:

Automating pixel-wise segmentation of skin lesions in RGB images using deep learning,
 allowing accurate identification of the lesion area in each image.

Key aspects of the challenge:

Segment lesions that can have highly variable shape, size, color, and texture.

Deal with image conditions such as varying lighting, scale,
 and backgrounds (different skin tones, hair, shadows).

Enable consistent and reproducible lesion area marking across large photo datasets.

Model Application
Medical Imaging: Identifies and outlines lesion boundaries in dermoscopic or standard camera images.

Patient Tracking: Enables automatic follow-up by monitoring size, shape, or color change over multiple time points.

AI-Assisted Diagnosis: Supports downstream tasks like malignancy classification,
 measurement of lesion evolution, or treatment impact assessment.

Research & Datasets: Can rapidly annotate datasets,
 accelerating dermatological studies and machine learning development.

"""