# Emotion Detection Project 😃😢

## 📖 Overview
This project implements a **facial emotion detection system** to classify images as **Happy** or **Sad** using deep learning.  
It leverages the **VGG16 convolutional neural network**, fine-tuned for binary classification, with comprehensive data preprocessing and augmentation for robust performance.  

The dataset is split into **Train, Validation, and Test** sets, with images processed to focus on facial features.

---

## 📂 Project Structure

### Dataset Preparation
- Creates a `TVT` directory with `Train`, `Validation`, and `Test` subfolders, each containing `Happy` and `Sad` subdirectories.
- Splits dataset: **70% Train, 15% Validation, 15% Test**.

### Preprocessing
- Converts images to **RGB** to handle ICC profiles.
- Renames files for consistency (e.g., `Happy_0.jpg`, `Sad_1.jpg`).
- Crops faces using the **MTCNN** face detection library.
- Resizes images to **224x224** and normalizes pixel values to `[0,1]`.
- Applies **data augmentation** (rotation, zoom, flipping) to training data.

### Model Training
- Uses **pre-trained VGG16** (without top layers) with custom classifier layers.
- Freezes VGG16 layers to leverage pre-trained weights.
- Trains for **20 epochs** with Adam optimizer and sparse categorical crossentropy loss.
- Achieves **~82.54% test accuracy**.

### Testing
- Evaluates model performance on the **Test folder**.
- Supports **single-image testing** with face detection and emotion prediction.

---

## 🛠 Requirements

- Python 3.10+
- TensorFlow  
- OpenCV (cv2)  
- Pillow (PIL)  
- NumPy  
- MTCNN  

Install dependencies:
```bash
pip install tensorflow opencv-python pillow numpy mtcnn

## 📁 Directory Structure
