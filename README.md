# Support Vector Machines and Ensemble Learning

## Overview
This project implements Support Vector Machines (SVM) and ensemble learning techniques (Bagging and AdaBoost) from scratch. It is based on an academic assignment and focuses on the theoretical and practical applications of these models.

The project is divided into two main parts:
1. Implementation and analysis of SVMs with different kernels (Linear and RBF) and configurations (Hard and Soft margins).
2. Implementation of Bagging and AdaBoost for ensemble learning.

The project includes hyperparameter tuning using Grid Search and Randomized Search, along with performance evaluation metrics such as accuracy, F1-score, and misclassification analysis.

---

## Features
- **SVM Implementation**: Linear and RBF kernels with hard and soft margins using quadratic programming.
- **Ensemble Methods**:
  - Bagging with decision trees.
  - AdaBoost with decision trees.
- **Custom Hyperparameter Tuning**: Grid Search and Randomized Search.
- **Data Analysis and Visualization**: Misclassified instances and support vector analysis.
- **Auto-grading Compatibility**: Structured to comply with academic grading scripts.

---

## File Structure
| File               | Description                                                                                                                                      |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `svm.py`           | Implementation of Support Vector Machines.                                                                                                     |
| `ensembling.py`    | Implementation of ensemble learning methods (Bagging and AdaBoost).                                                                             |
| `utils.py`         | Utility functions for data preprocessing and common operations.                                                                                 |
| `config.py`        | Configuration file containing hyperparameters and validation scores for experiments.                                                            |
| `analysis.py`      | Scripts for analyzing the results, generating plots, and visualizing misclassified instances.                                                   |
| `test.py`          | Test scripts for debugging and verifying correctness.                                                                                           |
| `Assignment 3.pdf` | Assignment details and problem statement.                                                                                                       |
| `report.pdf`       | Detailed report of the analysis, results, and performance metrics.                                                                              |

---

## Dataset
The dataset required for this project is stored in the following Google Drive folder:
[Dataset Link](https://drive.google.com/drive/folders/1pMpXGxO0TQIXljM0wULGWjbLwuowe-We)

The dataset is structured into `Train`, `Val`, and `Test` folders, with subfolders for each class label. Positive and negative samples are chosen based on the logic outlined in the assignment (e.g., modulo arithmetic on the last digit of the entry number).

---

## Results
Detailed results, including metrics like F1-score, accuracy, and the number of support vectors, are documented in `report.pdf`. The top support vectors and misclassified instances are visualized in the analysis.
