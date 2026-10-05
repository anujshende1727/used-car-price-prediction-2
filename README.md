# 🚗 Used Car Price Prediction & Market Segmentation

## 📌 Project Overview

Used Car Price Prediction & Market Segmentation is an AI/ML-based application that predicts the estimated market price of a used car and analyzes different segments of the used-car market.

The project combines supervised and unsupervised machine learning techniques:

- Regression for used-car price prediction
- K-Means clustering for market segmentation
- PCA for dimensionality-reduction visualization
- Streamlit for the interactive web application

---

## 🎯 Objectives

1. Predict the estimated price of a used car.
2. Analyze used-car prices based on different attributes.
3. Identify different groups of cars using K-Means clustering.
4. Analyze the characteristics of different market segments.
5. Compare multiple machine-learning regression models.
6. Develop an interactive application using Streamlit.

---

## 📊 Dataset

The dataset contains information about used cars including:

- Year
- Brand
- Full model name
- Model name
- Price
- Distance travelled
- Fuel type
- City

### Dataset Size

- Total records: 1,725
- Features after removing ID columns: 8
- Brands: 31
- Cities: 15

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

### Application Framework
- Streamlit

### Development Environment
- Google Colab
- Visual Studio Code

---

## 🧠 Machine Learning Techniques

### 1. Regression

The following regression algorithms were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Random Forest was selected as the final prediction model.

### Final Random Forest Performance

- R² Score: approximately 0.722
- MAE: approximately ₹3.15 lakh
- RMSE: approximately ₹9.66 lakh

---

## 🔧 Feature Engineering

Additional features were created to improve the prediction model:

### Car Age

```text
Car Age = Reference Year - Manufacturing Year