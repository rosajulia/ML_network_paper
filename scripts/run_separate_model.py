from scripts.data_loader import load_data, split_data
from scripts.preprocessing import create_preprocessor
from scripts.models import create_pipelines
from scripts.evaluation import save_variables_final_model
from scripts.train_test_final_model import traintest_finalmodel

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

plt.switch_backend('agg')  # Prevents GUI errors on headless servers

from sklearn.metrics import roc_auc_score, recall_score
from sklearn.utils import resample

# Set target and feature set for missing results
target = ''
feature_set = '' 

# Load data and split into train and test sets
df = load_data("data/PredVar_Clin.xlsx", "data/PredVar_Con.xlsx", target)
X_train, X_test, y_train, y_test, features, target = split_data(df, target, feature_set)

# Create preprocessing pipeline and model pipelines
pipelines = create_pipelines(create_preprocessor(feature_set))

# Put here model name + parameters, see example below
model_name = "Logistic Regression"
parameter_set = {
    'lg__C': 0.01,
    'lg__max_iter': 1000,
    'lg__penalty': 'l2',
    'lg__solver': 'lbfgs'
}

# Train final model on the full training set and evaluate on test set
# Ensure you use the right pipeline
final_alg = pipelines['pipe_LR'].set_params(**parameter_set)
_, _, _, _, _ = traintest_finalmodel(X_train, y_train, X_test, y_test, final_alg, target, feature_set, n_bootstrap=1000)
