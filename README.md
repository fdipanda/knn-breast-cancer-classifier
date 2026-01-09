# knn-breast-cancer-classifier
A C# implementation of the k-Nearest Neighbors classifier applied to the Wisconsin Breast Cancer dataset, including normalization and evaluation.

## Overview
This project implements a **k-Nearest Neighbors (k-NN) classifier from scratch** in C# and applies it to the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

The implementation avoids machine learning libraries and focuses on core algorithmic concepts such as distance computation, feature normalization, and classification evaluation.

## Algorithms Implemented
- k-Nearest Neighbors (k-NN)
- Euclidean distance metric
- Majority voting
- Min–max feature normalization

## Technologies Used
- **Language:** C#
- **Domain:** Machine Learning / Pattern Recognition
- **Concepts:** Supervised learning, distance-based classification, evaluation metrics

## Dataset
- **Wisconsin Diagnostic Breast Cancer (WDBC)**
- 30 numerical features per sample
- Binary classification labels (benign / malignant)

## How It Works
1. The dataset is loaded from a CSV file
2. Features are normalized using min–max scaling
3. Data is randomly shuffled and split into training (70%) and test (30%) sets
4. For each test sample:
   - Distances to all training samples are computed
   - The `k` nearest neighbors are selected
   - The predicted label is determined by majority vote
5. Accuracy and a confusion matrix are computed

## How to Run
```bash
dotnet run
```

Ensure `wdbc.data.mb.csv` is present in the project directory before running.

## Evaluation
The classifier is evaluated using multiple values of `k`:
- k = 1
- k = 3
- k = 5
- k = 7
- k = 9

For each value, the program reports:
- Classification accuracy
- Confusion matrix

## Academic Context
This project was developed to practice:
- Supervised machine learning algorithms
- Distance-based classification
- Data normalization
- Model evaluation using confusion matrices

## Author
Franck Dipanda
