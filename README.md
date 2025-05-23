# 🫀 Heart Failure Detection with Neural Networks

This project is a machine learning solution aimed at predicting heart failure events using clinical data. It uses a neural network model to analyze patient features and classify whether a heart failure event is likely to occur. The ultimate goal is to support healthcare professionals with early and accurate predictions.

---

## 🔍 Project Objective

Build and train a neural network to predict the binary outcome (`DEATH_EVENT`) of patients based on various clinical features. This classification model can help in early diagnosis and treatment planning.

---

## 🧠 Dataset Overview

The dataset consists of **299 patient records** with 13 features each. The target variable is `DEATH_EVENT` (1 = event occurred, 0 = survived).

### Selected Features:
- Age
- Anaemia
- Creatinine phosphokinase
- Diabetes
- Ejection fraction
- High blood pressure
- Platelets
- Serum creatinine
- Serum sodium
- Sex
- Smoking

### Dropped:
- `time` (not useful for prediction)

---

## 🏗️ Model Architecture

The neural network has the following layers:

- Input layer with 11 standardized features
- Hidden layers with:
  - ReLU activation
  - Batch Normalization
  - Dropout (for regularization)
  - L2 Regularization
- Output layer with sigmoid activation (binary output)

### Training Configuration:
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy and AUC
- Technique: Early stopping based on validation loss

---

## 📊 Model Evaluation

The model was evaluated using:
- Accuracy on test data
- AUC (Area Under the Curve)
- Classification report
- Confusion matrix
- Training and validation curves for accuracy and loss

---

## 📁 Output Files

- `heart_failure_nn_model.h5` – Trained model saved in HDF5 format.
- `scaler.save` – StandardScaler object saved with Joblib for reuse in inference.

These can be used later for real-time predictions or integration into applications.

---

## 📦 Libraries Used

- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib

---

## 📚 References

- [Heart Failure Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
- [Related Research Paper](https://doi.org/10.1016/j.jbi.2020.103369)

---

## 👤 Author

**Uta** – Student passionate about programming, robotics, and AI-powered healthcare solutions. Constantly exploring, building, and learning.

---

## 🌟 Future Improvements

- Add model interpretability (e.g., SHAP, LIME)
- Compare performance with traditional ML models like Random Forest, SVM, or XGBoost
- Deploy the model into a web or mobile app for practical use
- Collect more data for further training and evaluation

---
