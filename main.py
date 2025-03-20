import argparse
import time
import numpy as np
import pandas as pd
from scipy import interpolate

# Third-Party Libraries
from imblearn.combine import SMOTETomek

# Project Modules
from scripts.data_loader import load_data, split_data
from scripts.preprocessing import create_preprocessor
from scripts.models import create_pipelines, set_hyperparams
from scripts.cross_validation_methods import nestedCV
from scripts.evaluation import (
    save_variables, roc_auc_plot, create_shap_plots, prec_recall_plot, save_variables_final_model
)
from scripts.train_test_final_model import traintest_finalmodel

"""
main.py - Cognitive Impairment Classification Pipeline

This script runs a machine learning pipeline to classify cognitive impairment (CI) in multiple sclerosis (MS) patients.
It includes data preprocessing, model training using nested cross-validation, and evaluation using ROC-AUC and SHAP.

Usage:
    python main.py -fs <feature_set> -t <target>

Arguments:
    -fs, --feature_set  : Feature set to use (Base, Base+Network, Base+Network+MRI, Base+Network+Clin)
    -t, --target        : Target variable (SDMT_2SD or CI-CP)

Author: Julia Jelgerhuis
Last changed: 20 March 2025
"""

def parse_arguments():
    """Parses command-line arguments for feature set and target variable."""
    parser = argparse.ArgumentParser(
        description="Arguments for cognitive classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-fs", "--feature_set", required=True,
        choices=['Base', 'Base+Network', "Base+Network+MRI", "Base+Network+Clin"],
        help="Determine the feature set."
    )
    parser.add_argument(
        "-t", "--target", required=True,
        choices=['SDMT_2SD', "CI-CP"],
        help="Determine the target variable."
    )
    return parser.parse_args()

def train_and_evaluate(X_train, y_train, X_test, y_test, feature_set, target):
    """Performs nested cross-validation, evaluates models, and saves results."""
    
    # Initialize models
    pipelines = create_pipelines(create_preprocessor(feature_set))
    
    # Handle class imbalance using SMOTETomek
    smt = SMOTETomek(random_state=50)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
    
    print(f"Training set after SMOTE-TK: CP (0): {np.sum(y_train_resampled == 0)}, CI (1): {np.sum(y_train_resampled == 1)}")
    
    # Nested Cross-Validation
    hyperparams = set_hyperparams()
    results, tprs, precs, shap_values = nestedCV(
        X_train, y_train, n_folds_outer=10, n_folds_inner=5, pipeline=pipelines,
        hyperparams=hyperparams.values(), scoring={"AUC": "roc_auc"},
        names=hyperparams.keys(), repetitions=100
    )
    
    # Save results
    save_variables(results, tprs, shap_values, target, feature_set, "Best_Model", 0.81, {})  # Example placeholder
    print("Training & Evaluation Completed.")

def main():
    """Main execution pipeline for cognitive classification."""
    start_time = time.time()

    # Parse Arguments
    args = parse_arguments()
    
    # Load and preprocess data
    df = load_data("./data/PredVar_Clin.xlsx", "./data/PredVar_Con.xlsx", args.target)
    X_train, X_test, y_train, y_test, features, target = split_data(df, args.target, args.feature_set)

    print(f"Dataset Loaded - Feature Set: {args.feature_set}, Target: {args.target}")
    print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")
    
    # Train and evaluate models
    train_and_evaluate(X_train, y_train, X_test, y_test, args.feature_set, args.target)
    
    print(f"Total Runtime: {round((time.time() - start_time) / 60, 2)} minutes.")

if __name__ == "__main__":
    main()
