# Pneumonia-Classification-Project
Using Python technology and deep learning, I classified pneumonia in the Chest X-ray image.

## 1.Project Overview
This project focuses on classifying pneumonia cases from chest X-ray images using deep learning techniques. The dataset is sourced from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). The main goal is to classify images into pneumonia-positive (1) or pneumonia-negative (0) categories using convolutional neural networks (CNNs).

## 2.Project Structure
```
Pneumonia_Classification_Project/
│── notebooks/                      # Jupyter Notebooks
│   ├── pneumonia_classification.ipynb  # Main Jupyter Notebook
│   ├── Pneumonia_Project/               # Project-related data and scripts
│   │   ├── data/                         # Dataset files
│   │   │   ├── stage_2_train_labels.csv  # Training labels
│   │   │   ├── stage_2_train_images/     # X-ray image files
│   │   ├── processed_images/             # Preprocessed image files
│── src/                          # Source code for preprocessing and training
│── models/                       # Saved model weights
│── results/                       # Output graphs and metrics
│── README.md                      # Project documentation
```

## 3.Dataset and Preprocessing
### Dataset Used
- **stage_2_train_labels.csv**: Contains labels for training images.
- **stage_2_train_images/**: Directory containing X-ray images.

### Preprocessing Steps
- Load and visualize DICOM images using `pydicom`.
- Normalize pixel values for better model performance.
- Augment training data using rotation, flipping, and contrast adjustments.
- Split the dataset into 80% training/validation and 20% testing.

## 4.Model Architecture
- Implemented a convolutional neural network (CNN) using `tensorflow.keras`.
- **Model Layers**:
  - **Conv2D** layers with ReLU activation for feature extraction.
  - **MaxPooling2D** layers for dimensionality reduction.
  - **Dropout** layers to prevent overfitting.
  - **Dense** layers for final classification.
- **Loss Function**: Binary cross-entropy.
- **Optimizer**: Adam.

## 5.Training and Validation
### Training
The model was trained using the following configuration:
```python
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen)
)
```
- The dataset was processed using `train_gen` (training data generator) and `val_gen` (validation data generator).
- The model was trained for **10 epochs**, with batch-wise updates to optimize convergence.
- Each epoch processed all training batches (`steps_per_epoch=len(train_gen)`).
- Validation was performed after each epoch using `validation_steps=len(val_gen)`.
- Training accuracy improved progressively, and loss steadily decreased.

### Validation
- 20% of the training set was used for validation.
- Hyperparameters were adjusted based on validation performance.
- **Final validation accuracy**: **78.23%**

## 6.Results and Analysis
- **Final Test Accuracy**: **78.96%**
- Evaluated model performance on the test dataset.
- Plotted training and validation loss/accuracy curves.
- **Loss and Accuracy Trends**:
  - Training accuracy steadily improved over epochs.
  - Validation accuracy reached **78.23%**, indicating good generalization.
  - Loss consistently decreased, confirming effective learning.

## 7.Conclusion
- Successfully classified pneumonia cases from X-ray images using deep learning.
- Future improvements:
  - Experiment with different CNN architectures (e.g., ResNet, EfficientNet).
  - Fine-tune hyperparameters for better generalization.
  - Increase dataset diversity through additional augmentation techniques.

This project demonstrates the application of deep learning in medical imaging and serves as a portfolio that demonstrates exploration in the fields of data science, AI, and medical analytics.

