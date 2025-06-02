# BBM409 - KNN Classification & Regression Assignment

This repository contains my implementation for Assignment 1 of **BBM409: Introduction to Machine Learning Lab**, completed during Fall 2022 at Hacettepe University.

## 🧠 Assignment Overview

The project consists of two main parts:

### 1️⃣ Personality Classification (KNN Classification)

- **Dataset**: 60K samples with 60 features and 16 personality types
- **Algorithm**: K-Nearest Neighbors (k = 1, 3, 5, 7, 9)
- **Metrics**: Accuracy, Precision, Recall
- **Techniques**:
  - 5-fold cross-validation
  - Feature normalization (min-max scaling)
  - Weighted & unweighted KNN

### 2️⃣ Energy Efficiency Estimation (KNN Regression)

- **Dataset**: 768 samples describing building features and their energy loads
- **Targets**: Heating Load & Cooling Load
- **Metric**: Mean Absolute Error (MAE)
- **Techniques**:
  - 5-fold cross-validation
  - Feature normalization
  - Weighted & unweighted regression

## 📁 File Structure

- `code.py`: Full implementation of classification and regression models
- `report.ipynb`: Analysis, result tables, and error discussion
- `Assignment1_Fall2022_409.pdf`: Original assignment description

## ⚙️ Technologies

- Python
- NumPy, Pandas
- No external ML libraries used (all algorithms are implemented from scratch)

## 🚀 How to Run

Make sure you have the required CSV datasets:
- `subset_16P.csv`
- `energy_efficiency_data.csv`

Then run:
```bash
python code.py
