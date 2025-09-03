# Emotion Detection Project

Overview

This project implements a facial emotion detection system to classify images as "Happy" or "Sad" using deep learning. It leverages the VGG16 convolutional neural network, fine-tuned for binary classification, and includes comprehensive data preprocessing and augmentation to ensure robust performance. The dataset is organized into Train, Validation, and Test splits, with images processed to focus on facial features.

Project Structure

The project is organized into the following key components:





Dataset Preparation:





Creates a TVT directory with Train, Validation, and Test subfolders, each containing Happy and Sad subdirectories.



Splits the dataset into 70% Train, 15% Validation, and 15% Test.



Preprocessing:





Handles ICC profiles by converting images to RGB format.



Renames files for consistency (e.g., Happy_0.jpg, Sad_1.jpg).



Crops images to focus on faces using the MTCNN face detection library.



Resizes images to 224x224 pixels and normalizes pixel values to [0,1].



Applies data augmentation (rotation, zoom, flipping) to enhance training data diversity.



Model Training:





Uses a pre-trained VGG16 model (without top layers) as the base, with custom fully connected layers for classification.



Freezes VGG16 layers to leverage pre-trained weights, training only the custom classifier.



Trains the model for 20 epochs using the Adam optimizer and sparse categorical crossentropy loss.



Achieves a test accuracy of approximately 82.54%.



Testing:





Evaluates the model on the entire Test folder, reporting accuracy.



Supports single-image testing with face detection and emotion prediction.

Requirements

To run this project, install the following dependencies:





Python 3.10+



TensorFlow



OpenCV (cv2)



Pillow (PIL)



NumPy



MTCNN



Keras

Install them using:

pip install tensorflow opencv-python pillow numpy mtcnn

Directory Structure

project_root/
│
├── data/
│   ├── Happy/
│   ├── Sad/
│
├── TVT/
│   ├── Train/
│   │   ├── Happy/
│   │   ├── Sad/
│   ├── Validation/
│   │   ├── Happy/
│   │   ├── Sad/
│   ├── Test/
│   │   ├── Happy/
│   │   ├── Sad/
│
├── TestImage/
│   ├── 333.jpg  # Example image for single-image testing
│
├── vgg16_emotion_model.h5  # Trained model
├── README.md
├── notebook.ipynb  # Jupyter notebook with all code

Usage





Prepare the Dataset:





Place your Happy and Sad image datasets in the data/Happy and data/Sad folders.



Run the dataset preparation script to create the TVT directory and split the data.



Preprocess the Data:





Execute the preprocessing steps to handle ICC profiles, rename files, crop faces, resize, and normalize images.



Apply data augmentation to the training set.



Train the Model:





Run the model training script to fine-tune the VGG16 model.



The trained model is saved as vgg16_emotion_model.h5.



Test the Model:





Evaluate the model on the Test folder to compute accuracy.



Test a specific image (e.g., 333.jpg in TestImage/) by running the single-image testing script.

Example for single-image testing:

img_name = "333.jpg"
folderDir = os.path.join(os.getcwd(), "TestImage")
# Ensure the model and image paths are correct in the script

Results





Test Accuracy: ~82.54% on the Test dataset.



Model Performance: The model generalizes well due to data augmentation and face cropping, though further tuning (e.g., unfreezing VGG16 layers) could improve accuracy.



Output: For single-image testing, the model outputs "Happy" or "Sad" based on the detected face.

Future Improvements





Unfreeze some VGG16 layers for fine-tuning to improve accuracy.



Experiment with other architectures like ResNet or EfficientNet.



Expand the dataset with more diverse images.



Add support for additional emotions (e.g., Angry, Neutral).

Contributing

Feel free to fork this repository, submit issues, or create pull requests. Contributions to improve the model, add features, or enhance documentation are welcome!

License

This project is licensed under the MIT License.
