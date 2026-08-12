# SteelSight AI: Steel Plate Fault Classification

## Student Information

**Name:** Madhan Gopal H R
**BITS ID:** 2025ac05909
**Course:** Machine Learning
**Assignment:** Assignment 2

## A. Problem Statement

Steel plates used in manufacturing can develop different surface faults that
affect product quality and structural reliability. Manual fault identification
can be time-consuming and may produce inconsistent results.

This project develops a multiclass machine-learning system to classify steel
plate surface faults using geometric, luminosity, thickness and shape-related
measurements. Five classification models are trained and evaluated on the same
dataset. The trained models are integrated into an interactive Streamlit web
application that allows users to upload test data, select a model and inspect
its evaluation results.

The seven fault categories considered are:

1. Bumps
2. Dirtiness
3. K_Scatch
4. Other_Faults
5. Pastry
6. Stains
7. Z_Scratch

## B. Dataset Description

The project uses the **Steel Plates Faults dataset** from the UCI Machine
Learning Repository.

**Dataset source:**
https://archive.ics.uci.edu/dataset/198/steel+plates+faults

**Dataset DOI:**
https://doi.org/10.24432/C5J88N

### Dataset Summary

| Property | Description |
|---|---|
| Number of instances | 1,941 |
| Number of input features | 27 |
| Number of target classes | 7 |
| Problem type | Multiclass classification |
| Missing values | None |
| Duplicate records | None |
| Feature types | Integer and continuous |
| Test-set size | 389 records |
| Train-test split | 80:20 stratified split |

The original dataset represents the output through seven one-hot-encoded
columns. These columns were converted into one categorical target named
`Fault_Type`.

The dataset is imbalanced. `Other_Faults` is the largest class, while
`Dirtiness` is the smallest class. Therefore, stratified sampling was used to
preserve class proportions in the training and test sets.

## C. GitHub Repository Link

**GitHub Repository:**
https://github.com/2025ac05909/steel-plate-fault-classification

## D. Models Used and Evaluation Results

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbours
4. Gaussian Naive Bayes
5. Random Forest Classifier

All models were evaluated on the same stratified test dataset.

For multiclass evaluation:

- Precision, Recall and F1-score use weighted averaging.
- AUC uses the weighted One-vs-Rest approach.
- MCC is included because it provides a balanced evaluation for multiclass and
  imbalanced classification problems.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7275 | 0.9066 | 0.7329 | 0.7275 | 0.7277 | 0.6487 |
| Decision Tree | 0.7455 | 0.8313 | 0.7492 | 0.7455 | 0.7452 | 0.6733 |
| K-Nearest Neighbours | 0.7275 | 0.8941 | 0.7318 | 0.7275 | 0.7245 | 0.6548 |
| Gaussian Naive Bayes | 0.4524 | 0.7955 | 0.5741 | 0.4524 | 0.4003 | 0.3789 |
| Random Forest | **0.8021** | **0.9468** | **0.8087** | **0.8021** | **0.8016** | **0.7434** |

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved 72.75% accuracy and a strong AUC of 0.9066. It performed well for K_Scatch, Stains and Z_Scratch but showed confusion among Bumps, Other_Faults and Pastry. |
| Decision Tree | The Decision Tree achieved 74.55% accuracy and performed better than Logistic Regression in Accuracy, F1 and MCC. Its lower AUC of 0.8313 indicates less reliable probability-based class separation. |
| K-Nearest Neighbours | KNN achieved 72.75% accuracy and an AUC of 0.8941. Feature scaling improved the distance calculations, but the model had difficulty distinguishing Other_Faults and Pastry from Bumps. |
| Gaussian Naive Bayes | Gaussian Naive Bayes produced the lowest performance, with 45.24% accuracy and an F1-score of 0.4003. Its conditional-independence and Gaussian-distribution assumptions were unsuitable for the correlated steel-plate measurements. |
| Random Forest | Random Forest achieved the best values for all six evaluation metrics. Its ensemble of decision trees captured nonlinear feature relationships and reduced the variance associated with an individual tree. |
| Overall Winner | **Random Forest** is the overall winner because it obtained the highest Accuracy, AUC, Precision, Recall, F1 and MCC on the test dataset. |

## Streamlit Application Features

The SteelSight AI Streamlit application provides:

- CSV test-data upload
- Classification-model selection
- Dataset summary
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Multiclass confusion matrix
- Detailed classification report
- Actual and predicted fault-class comparison
- Downloadable prediction results

## Repository Structure

```text
steel-plate-fault-classification/
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
├── model_comparison.csv
├── ML_Assignment_2.ipynb
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── gaussian_naive_bayes.pkl
    ├── random_forest.pkl
    ├── label_encoder.pkl
    └── feature_columns.pkl
