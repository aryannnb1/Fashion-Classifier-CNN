# Fashion Classifier using Convolutional Neural Networks (CNN)

## Overview

This project focuses on classifying garments using Convolutional Neural Networks (CNN). The project involves data preprocessing, model training, and evaluation to achieve accurate garment classification.

## Dataset

The dataset used for this project is the Fashion MNIST dataset, which contains 70,000 grayscale images of 10 different types of clothing items. Each image is 28x28 pixels in size. The dataset is divided into 60,000 training images and 10,000 test images. The 10 classes in the dataset are:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Data Preprocessing

### Normalization
- Scaled pixel values to the range [0, 1].

### Reshaping
- Reshaped the data to fit the input requirements of the CNN model.

### One-Hot Encoding
- Converted the class labels to one-hot encoded vectors.

## Model Architecture

The CNN model consists of the following layers:

1. **Conv2D**: 32 filters, kernel size 3x3, ReLU activation
2. **MaxPooling2D**: Pool size 2x2
3. **Conv2D**: 64 filters, kernel size 3x3, ReLU activation
4. **MaxPooling2D**: Pool size 2x2
5. **Flatten**
6. **Dense**: 128 units, ReLU activation
7. **Dropout**: 0.5
8. **Dense**: 10 units, Softmax activation

## Model Training

- Compiled the model using Adam optimizer and categorical cross-entropy loss function.
- Trained the model on the training dataset.
- Evaluated the model performance using the test dataset.

## Evaluation Metrics

- **Training Accuracy**: 99.12%
- **Test Accuracy**: 90.45%
- **Loss**: Evaluated using categorical cross-entropy

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Jupyter Notebook or any Python IDE

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aryannnb1/Fashion-Classifier-CNN.git
   cd Fashion-Classifier-CNN
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Run the data preprocessing and model training notebook to clean and analyze the dataset.
3. Run the evaluation notebook to test the model performance.

## Results

The final model performance will be evaluated using accuracy and loss metrics to determine the accuracy and reliability of the predictions.
