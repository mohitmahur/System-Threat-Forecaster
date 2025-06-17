# 🛡️ System Threat Forecaster - Malware Infection Prediction

This project is part of a machine learning competition to predict a system’s probability of getting infected by various families of malware. The data consists of telemetry properties collected from antivirus software logs.

---

## 🚀 Problem Statement

The goal of this project is to build a machine learning model that can **predict the probability of malware infection** based on system properties. The telemetry data includes various system configurations, usage patterns, and security settings.

---

## 📊 Dataset

The dataset includes anonymized telemetry data collected from systems. Each row corresponds to a system, and the target is a label indicating infection by a specific malware family.

---

## 🧠 ML Techniques Used

- **Data Preprocessing**:
  - Handling missing values
  - Encoding categorical variables
  - Scaling numerical features
  - Variance filtering

- **Class Imbalance Handling**:
  - SMOTE (Synthetic Minority Oversampling Technique)

- **Modeling**:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - XGBoost
  - Ridge, Lasso, SGD Classifiers
  - SVM
  - Ensemble models: Bagging, AdaBoost, Gradient Boosting, Stacking

- **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

## 📚 Libraries Used

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from IPython.display import display, HTML
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
import warnings
