# Medical-Image-Classifier-Model

Here's a `README.md` file for your medical image classification project:

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify medical images into three distinct categories: X-ray, MRI, and CT scans. The model is trained from scratch on a custom dataset of these image types.

## Project Overview

The primary objective of this project is to develop an automated system that can accurately identify the modality of a given medical image (X-ray, MRI, or CT scan). This capability can be useful in various medical imaging workflows, such as organizing large datasets, routing images to appropriate specialists, or as a pre-processing step for more specialized analyses.

## Features

  - **Multi-class Classification:** Distinguishes between X-ray, MRI, and CT scan images.
  - **Custom CNN Architecture:** A simple yet effective CNN built from scratch for this classification task.
  - **Data Augmentation:** Basic image rescaling and automatic train/validation split for robust training.
  - **Training Visualization:** Plots for training and validation accuracy and loss over epochs.
  - **Model Saving:** Saves the trained model for future inference.


## Dataset Preparation

The script expects your medical image dataset to be organized into a single root directory (specified by `data_dir = "dataset"` in the code), with subdirectories for each image modality.

Your directory structure should look like this:

```
dataset/
├── CT/
│   ├── ct_scan_1.png
│   ├── ct_scan_2.png
│   └── ...
├── MRI/
│   ├── mri_scan_1.png
│   ├── mri_scan_2.png
│   └── ...
└── XRAY/
    ├── xray_scan_1.png
    ├── xray_scan_2.png
    └── ...
```

  - Replace `CT`, `MRI`, and `XRAY` with the actual names of your class directories if they differ.
  - The `ImageDataGenerator` automatically splits the data into training and validation sets based on `validation_split=0.2`.

## The script will:

1.  Set up the data generators.
2.  Define and compile the CNN model.
3.  Train the model for 10 epochs.
4.  Save the trained model as `medical_image_classifier.h5`.
5.  Display plots showing the training and validation accuracy and loss.

## Model Architecture

The CNN model is a sequential model designed for image classification:

  - **Input Layer:** Expects images of size $(224, 224, 3)$ (height, width, color channels).
  - **Convolutional Blocks:**
      - `Conv2D` with 32 filters, $(3, 3)$ kernel, `relu` activation, followed by `MaxPooling2D` $(2, 2)$.
      - `Conv2D` with 64 filters, $(3, 3)$ kernel, `relu` activation, followed by `MaxPooling2D` $(2, 2)$.
      - `Conv2D` with 128 filters, $(3, 3)$ kernel, `relu` activation, followed by `MaxPooling2D` $(2, 2)$.
  - **Flatten Layer:** Converts the 3D feature maps into a 1D vector.
  - **Dense Layers:**
      - A `Dense` layer with 128 units and `relu` activation.
      - A `Dropout` layer with a rate of 0.5 for regularization.
      - A final `Dense` layer with 3 units (corresponding to the three classes: X-ray, MRI, CT) and `softmax` activation for multi-class probability distribution.

## Training Details

  - **Image Dimensions:** $224 \\times 224$ pixels.
  - **Batch Size:** 32.
  - **Optimizer:** Adam optimizer with a learning rate of $0.001$.
  - **Loss Function:** Categorical Crossentropy (`categorical_crossentropy`), suitable for multi-class classification.
  - **Metrics:** Accuracy.
  - **Epochs:** 10.
  - **Data Split:** 80% for training, 20% for validation.

## Results

Upon completion of training, the model will be saved, and two plots will be displayed:

  - **Training and Validation Accuracy:** Shows how well the model performed on both sets over epochs.
  - **Training and Validation Loss:** Illustrates the error rate during training and validation.

These plots provide insights into model performance and help identify issues like overfitting or underfitting.

## Future Enhancements

  - **Data Augmentation:** Implement more advanced augmentation techniques (e.g., rotation, shifts, zoom) to improve generalization.
  - **Transfer Learning:** Experiment with pre-trained models (e.g., ResNet, VGG) for potentially higher accuracy and faster convergence, especially with larger datasets.
  - **Hyperparameter Tuning:** Optimize learning rate, batch size, and network architecture parameters.
  - **Model Evaluation:** Implement more comprehensive evaluation metrics beyond accuracy, such as precision, recall, F1-score, and confusion matrices.
  - **Inference Script:** Create a separate script to load the saved model and perform predictions on new, unseen medical images.
  - **Dataset Expansion:** Train on a larger and more diverse dataset for improved robustness.

## Contact

For any questions or inquiries, please contact me at alanvarghese852@gmail.com.

-----
