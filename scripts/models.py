from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

"""
models.py - Model Creation and Hyperparameter Configuration

This module defines and configures machine learning models for cognitive impairment classification.
It includes pipeline setup for preprocessing, resampling, and model training.

Functions:
    - create_pipelines: Creates ML pipelines with preprocessing and resampling steps.
    - set_hyperparams: Defines hyperparameter grids for model tuning.

Author: Julia Jelgerhuis
Last changed: 20 March 2025
"""

def create_pipelines(preprocessor):
    """
    Creates machine learning pipelines incorporating preprocessing and resampling.
    
    Args:
        preprocessor (sklearn ColumnTransformer): Preprocessing pipeline.
    
    Returns:
        dict: Dictionary containing model pipelines.
    """
    # Define models
    randfor = RandomForestClassifier(random_state=50)
    logreg = LogisticRegression(random_state=50, max_iter=1000)
    svm = SVC(probability=True, random_state=50)

    # Resampling strategy
    smt = SMOTETomek(random_state=50)

    # Pipelines for each model
    pipelines = {
        'pipe_RF': Pipeline([('preprocessor', preprocessor), ('smt', smt), ('rf', randfor)]),
        'pipe_LR': Pipeline([('preprocessor', preprocessor), ('smt', smt), ('lg', logreg)]),
        'pipe_svm': Pipeline([('preprocessor', preprocessor), ('smt', smt), ('svm', svm)])
    }
    
    return pipelines

def set_hyperparams():
    """
    Defines hyperparameter grids for tuning models.
    
    Returns:
        dict: Dictionary containing hyperparameter grids for each model.
    """
    hyperparams = {
        "RandomForest": {
            'rf__n_estimators': [50, 75, 100],
            'rf__max_features': ["sqrt", "log2"],
            'rf__max_depth': [5, 7, 9],
            'rf__max_samples': [0.3, 0.5, 0.8],
            'rf__criterion': ['entropy'],
            'rf__class_weight': ["balanced_subsample"]
        },
        'LogisticRegression': [
            {
                'lg__penalty': ['l2'],
                'lg__solver': ['lbfgs'],
                'lg__C': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            {
                'lg__penalty': ['l1', 'l2'],
                'lg__solver': ['liblinear'],
                'lg__C': [0.001, 0.01, 0.1, 1, 10, 100]
            }
        ],
        "SVM": {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': [1, 0.1, 0.01, 0.001],
            'svm__kernel': ['rbf', 'poly', 'sigmoid']
        }
    }
    return hyperparams
