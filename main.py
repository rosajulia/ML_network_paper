from scripts.data_loader import load_data, split_data
from scripts.preprocessing import create_preprocessor
from scripts.models import create_pipelines, set_hyperparams
from scripts.cross_validation_methods import nestedCV
from scripts.evaluation import (
    save_variables,
    roc_auc_plot,
    create_shap_plots,
    prec_recall_plot,
    save_variables_final_model,
)
from scripts.train_test_final_model import traintest_finalmodel

import numpy as np
import pandas as pd
import shap
import argparse
import time

from scipy import interpolate
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks


def interpolate_tprs(tprs, n_points=100):
    """
    Interpolates TPRs to a common set of thresholds.
    """
    mean_fpr = np.linspace(0, 1, n_points)
    interpolated_tprs = []

    for name, tpr in tprs:
        if len(tpr[0]) == 0:
            # Skip empty TPR arrays
            continue

        interp_tpr = interpolate.interp1d(
            np.linspace(0, 1, len(tpr[0])),
            tpr[0],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interpolated_tprs.append(interp_tpr(mean_fpr))

    return np.array(interpolated_tprs)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Arguments for cognitive classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-fs",
        "--feature_set",
        choices=[
            "Base", "Base+Network", "Base+Network+MRI", "Base+Network+Clin",
            "Base+FuncNetwork", "Base+StructNetwork", "Base+Clin", "Base+MRI"],
        help="Determine featureset. Choices are: Base, Base+Network, Base+Network+MRI, Base+Network+Clin,  \
        Base+FuncNetwork, Base+StructNetwork, Base+Clin, Base+MRI",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=["SDMT_2SD", "CI-CP"],
        help="Determine target. Choices are SDMT_2SD or CI-CP",
        required=True,
    )

    args = parser.parse_args()
    config = vars(args)
    print(config)

    #### PREPROCESSING ####

    # Load and preprocess data
    target = args.target
    feature_set = args.feature_set

    df = load_data("./data/PredVar_Clin.xlsx", "./data/PredVar_Con.xlsx", target)
    X_train, X_test, y_train, y_test, features, target = split_data(
        df, target, feature_set
    )

    print("Features: ", features)
    print("Target: ", target)
    print("X_train : ", X_train.shape)
    print("X_test : ", X_test.shape)
    print("y_train : ", y_train.shape)
    print("y_test : ", y_test.shape)

    # Counting the instances of the target in both sets
    unique, counts_train = np.unique(y_train, return_counts=True)
    train_counts = dict(zip(unique, counts_train))

    unique, counts_test = np.unique(y_test, return_counts=True)
    test_counts = dict(zip(unique, counts_test))

    # Outputting the counts for CP (0) and CI (1) in both sets
    print(
        f"Training set counts before SMOTE-TK (with stratified split): Target (0): "
        f"{train_counts.get(0, 0)}, Target (1): {train_counts.get(1, 0)}"
    )
    print(
        f"Test set counts (with stratified split): Target (0): "
        f"{test_counts.get(0, 0)}, Target (1): {test_counts.get(1, 0)}"
    )

    # Create modules
    pipelines = create_pipelines(create_preprocessor(feature_set))

    '''         DIAGNOSTICS, not needed for run
    # Outputting counts for CP (0) and CI (1) in training data after SMOTE-TK
    # Apply SMOTE-TK to the training data
    smt = SMOTETomek(random_state=50)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

    unique, counts_resampled = np.unique(y_train_resampled, return_counts=True)
    resampled_counts = dict(zip(unique, counts_resampled))

    print(
        f"Training set counts after SMOTE-TK: "
        f"CP (0): {resampled_counts.get(0, 0)}, "
        f"CI (1): {resampled_counts.get(1, 0)}"
    )
    '''
    # Set parameters for nested cross-validation
    # DRY-RUN
    #n_folds_inner = 2
    #n_folds_outer = 2
    #repetitions = 1

    n_folds_inner = 5
    n_folds_outer = 10
    repetitions = 100

    scoring = {"AUC": "roc_auc"}
    hyperparams = set_hyperparams()

    #### NESTED CV AND VISUALIZATION ####

    results, tprs, precs, shap_values = nestedCV(
        X_train,
        y_train,
        n_folds_inner,
        n_folds_outer,
        pipelines,
        hyperparams.values(),
        scoring,
        hyperparams.keys(),
        repetitions,
        start_repeat=start_repeat,
    )

    # Perform nested cross-validation
    results, tprs, precs, shap_values = nestedCV(
        X_train,
        y_train,
        n_folds_inner,
        n_folds_outer,
        pipelines,
        hyperparams.values(),
        scoring,
        hyperparams.keys(),
        repetitions
    )

    # Create roc auc plots of all algorithms from nested CV
    # ROC AUC plot
    logreg_tprs = interpolate_tprs(
        [row for row in tprs if row[0][0] == "LogisticRegression"]
    )
    randfor_tprs = interpolate_tprs(
        [row for row in tprs if row[0][0] == "RandomForest"]
    )
    svm_tprs = interpolate_tprs([row for row in tprs if row[0][0] == "SVM"])

    roc_auc_plot(logreg_tprs, "LR", target, feature_set)
    roc_auc_plot(randfor_tprs, "RF", target, feature_set)
    roc_auc_plot(svm_tprs, "SVM", target, feature_set)

    # Create plots
    logreg_dict = {}
    randfor_dict = {}
    svm_dict = {}

    # Shapley plot
    for subj, data in shap_values.items():
        logreg_dict[subj] = {}
        randfor_dict[subj] = {}
        svm_dict[subj] = {}

        for key, arrays in data.items():
            logreg_dict[subj][key] = pd.Series(
                arrays.get("LogisticRegression", np.zeros_like(X_train.iloc[0]))
            )
            randfor_dict[subj][key] = pd.Series(
                arrays.get("RandomForest", np.zeros_like(X_train.iloc[0]))
            )
            svm_dict[subj][key] = pd.Series(
                arrays.get("SVM", np.zeros_like(X_train.iloc[0]))
            )

    create_shap_plots(X_train, logreg_dict, "LR", target, feature_set)
    create_shap_plots(X_train, randfor_dict, "RF", target, feature_set)
    create_shap_plots(X_train, svm_dict, "SVM", target, feature_set)

    # Precision recall plots
    logreg_precs = [row[1][0] for row in precs if row[0][0] == "LogisticRegression"]
    randfor_precs = [row[1][0] for row in precs if row[0][0] == "RandomForest"]
    svm_precs = [row[1][0] for row in precs if row[0][0] == "SVM"]

    prec_recall_plot(logreg_precs, "LR", target, feature_set)
    prec_recall_plot(randfor_precs, "RF", target, feature_set)
    prec_recall_plot(svm_precs, "SVM", target, feature_set)

    #### FIND BEST ALGORITHM AND BEST HYPERPARAMETERS ####

    # Extract ROC AUC values for the specific algorithms
    logreg_rocauc = [row[3] for row in results[1:] if row[2] == "LogisticRegression"]
    randfor_rocauc = [row[3] for row in results[1:] if row[2] == "RandomForest"]
    svm_rocauc = [row[3] for row in results[1:] if row[2] == "SVM"]

    # Determine best model based on AUC
    logreg_mean_rocauc = np.mean(logreg_rocauc)
    randfor_mean_rocauc = np.mean(randfor_rocauc)
    svm_mean_rocauc = np.mean(svm_rocauc)

    # Find the algorithm with the highest mean ROC AUC
    max_mean_rocauc_algorithm = max(
        [
            ("LogisticRegression", logreg_mean_rocauc),
            ("RandomForest", randfor_mean_rocauc),
            ("SVM", svm_mean_rocauc),
        ],
        key=lambda x: x[1],
    )

    print(
        f"The algorithm with the highest mean ROC AUC is "
        f"{max_mean_rocauc_algorithm[0]} with a mean ROC AUC of "
        f"{max_mean_rocauc_algorithm[1]} train on feature set: "
        f"{feature_set} with target {target}"
    )

    # For the algorithm with the highest mean ROC AUC, find the best parameters (highest AUC)
    selected_algorithm = max_mean_rocauc_algorithm[0]
    selected_algorithm_data = [
        row for row in results[1:] if row[2] == selected_algorithm
    ]
    highest_rocauc_row = max(selected_algorithm_data, key=lambda x: x[3])

    # Extract ROC AUC and parameter set from the row with the highest ROC AUC
    highest_rocauc = highest_rocauc_row[3]
    parameter_set = highest_rocauc_row[9]

    # Save results
    save_variables(
        results,
        tprs,
        shap_values,
        target,
        feature_set,
        selected_algorithm,
        highest_rocauc,
        parameter_set,
    )

    #### TRAIN FINAL MODEL ON WHOLE TRAINING SET AND EVALUATE ON TEST SET ####

    if selected_algorithm == "LogisticRegression":
        final_alg = pipelines["pipe_LR"].set_params(**parameter_set)
    elif selected_algorithm == "RandomForest":
        final_alg = pipelines["pipe_RF"].set_params(**parameter_set)
    elif selected_algorithm == "SVM":
        final_alg = pipelines["pipe_svm"].set_params(**parameter_set)
    else:
        print("Name algorithm not recognized")
        exit()

    results, tprs, precs, shap_values, final_model = traintest_finalmodel(
        X_train,
        y_train,
        X_test,
        y_test,
        final_alg,
        target,
        feature_set,
    )

    shap.summary_plot(shap_values, X_test, show=False)

    # Save the rest of the variables
    save_variables_final_model(
        results,
        tprs,
        shap_values,
        target,
        feature_set,
        final_model,
    )

    print("Runtime in hours:", (time.time() - start_time) / 3600)


if __name__ == "__main__":
    main()
