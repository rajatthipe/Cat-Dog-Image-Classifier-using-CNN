# Cat-Dog-Image-Classifier-using-CNN

# 🐱🐶 CatDogNet

**CatDogNet** is a deep learning project built to classify images of cats and dogs using a Convolutional Neural Network (CNN). The model is trained using TensorFlow and Keras on a structured dataset of labeled images. It demonstrates effective use of image preprocessing, CNN design, batch normalization, and dropout regularization to improve performance and reduce overfitting.

---

## 📌 Project Objective

Build a binary image classifier that can distinguish between cats and dogs using a CNN architecture and evaluate its performance on unseen image data.

---

## 🧰 Tech Stack

- **Language:** Python
- **Framework:** TensorFlow, Keras
- **Tools:** Jupyter Notebook, NumPy, Matplotlib (for visualization)
- **Model Storage:** `.keras` format

---

## 📂 Dataset Overview

- Dataset is split into two directories:
  - `training_set`: 8000 images (cats & dogs)
  - `test_set`: 2000 images
- Image shape standardized to **64x64 pixels**
- Loaded using `ImageDataGenerator` with normalization

---

## 📊 Data Preprocessing

- All images are **rescaled (1./255)** for normalization.
- Data is structured in folders with labels inferred from folder names.
- Batch size: **32**
- Data augmentation not used in this version (could be an enhancement).

---

## 🧠 CNN Model Architecture

* The CNN includes four convolutional blocks followed by three dense layers and an output layer.

## ⚙️ Training Configuration
* Loss Function: Binary Crossentropy

* Optimizer: Adam

* Metrics: Accuracy

* Epochs: 10

* Validation: On a separate test set

* Output: cat_dog_classifier.keras

## 📈 Performance Summary
* Training Accuracy: ~98% (after 10 epochs)

* Validation Accuracy: ~85–87%

* Loss: Reduced consistently across epochs

* Observation: Minor overfitting started after 8th epoch, but handled well with dropout and batch normalization.

## 🔍 Key Findings & Insights
* Deeper CNN architecture significantly improved feature extraction over shallow networks.

* Batch Normalization after dense layers stabilized training and improved generalization.

* Dropout Regularization (0.1, 0.2) reduced overfitting on the training data.

* Using 64x64 resolution accelerated training but potentially limited the granularity of visual features.

* Binary classification with sigmoid and binary_crossentropy was ideal due to only two classes.




