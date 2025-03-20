from sklearn.model_selection import GridSearchCV, StratifiedKFold
import shap
import numpy as np
import signal
from scripts.evaluation import calc_scoring

"""
cross_validation_methods.py - Nested Cross-Validation for Cognitive Impairment Classification

This module implements nested cross-validation (CV) for hyperparameter tuning and model evaluation. It includes
handling timeouts during training to prevent indefinite execution.

Functions:
    - fit_with_timeout: Fits a model with a specified timeout limit.
    - nestedCV: Performs nested cross-validation with repeated iterations and SHAP value extraction.

Author: Julia Jelgerhuis
Last changed: 20 March 2025
"""

def fit_with_timeout(gcv_model_select, X_train, y_train, timeout):
    """Fits a model with a timeout to prevent indefinite execution."""
    def handler(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        gcv_model_select.fit(X_train, y_train)
        signal.alarm(0)
        return gcv_model_select.best_estimator_, gcv_model_select.best_params_
    except TimeoutError:
        return None, None

def nestedCV(X, y, n_folds_inner, n_folds_outer, pipeline, hyperparams, scoring, names, repetitions, timeout=600):
    """
    Performs nested cross-validation on data with a timeout mechanism.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        n_folds_inner (int): Inner CV folds.
        n_folds_outer (int): Outer CV folds.
        pipeline (dict): Dictionary of model pipelines.
        hyperparams (dict): Hyperparameter grids.
        scoring (dict): Scoring metrics.
        names (list): Model names.
        repetitions (int): Number of repeated CV iterations.
        timeout (int): Timeout per model fit.

    Returns:
        tuple: Outer fold results, TPRs, Precision-Recall metrics, SHAP values.
    """
    outer_fold_results = []
    tprs = []
    precs = []
    shap_values_per_cv = {sample: {CV_repeat: {} for CV_repeat in range(repetitions)} for sample in X.index}

    for CV_repeat in range(repetitions):
        print(f'\n------------ CV Repeat number: {CV_repeat}')
        skf_outer = StratifiedKFold(n_splits=n_folds_outer, shuffle=True, random_state=CV_repeat)
        skf_inner = StratifiedKFold(n_splits=n_folds_inner, shuffle=True, random_state=CV_repeat)

        for i, (train_index, test_index) in enumerate(skf_outer.split(X, y)):
            print(f'\n------ Fold Number: {i}')
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for est, pgrid, name in zip(pipeline.values(), hyperparams, names):
                print(f'\n------ Algorithm: {name}')
                gcv_model_select = GridSearchCV(
                    estimator=est,
                    param_grid=pgrid,
                    scoring=scoring,
                    n_jobs=10,
                    cv=skf_inner,
                    verbose=1,
                    refit="AUC"
                )
                
                best_model, best_params = fit_with_timeout(gcv_model_select, X_train, y_train, timeout)
                
                if best_model is None:
                    print(f"Timeout reached for fold {i} with algorithm {name}, skipping this fold.")
                    outer_fold_results.append([CV_repeat, i, name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None])
                    tprs.append([[name], [np.zeros_like(y_test)]] )
                    precs.append([[name], [np.zeros_like(y_test)]] )
                    continue
                
                print(f'\n Best parameters: {best_params}')
                prob_pred = best_model.predict_proba(X_test)[:, 1]
                pred = best_model.predict(X_test)
                
                fpr, tpr, tpr_array, roc_auc, precision, recall, pr_auc, prec_array, f1, specificity = \
                    calc_scoring(y_test, prob_pred, pred)
                
                outer_fold_results.append([CV_repeat, i, name, roc_auc, precision, recall, pr_auc, f1, specificity, best_params])
                tprs.append([[name], [tpr_array]])
                precs.append([[name], [prec_array]])
                
                print(f'\n AUC: {roc_auc}, Precision: {precision}, Recall: {recall}, F1: {f1}, Specificity: {specificity}')
                
                explainer = shap.Explainer(best_model.predict, X_train)
                shap_values = explainer(X_test)
                
                for j, test_ix in enumerate(test_index):
                    shap_alg = {name: shap_values[j]}
                    shap_values_per_cv[X.index[test_ix]][CV_repeat].update(shap_alg)
                
                del explainer, shap_values  # Free memory
    
    return outer_fold_results, tprs, precs, shap_values_per_cv
