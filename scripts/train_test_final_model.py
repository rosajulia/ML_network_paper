import shap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import resample
from scripts.evaluation import calc_scoring

"""
train_test_final_model.py - Training and Testing of Final Model

This module trains the final selected model on the full training dataset and evaluates it on the test set.
It includes bootstrapping for confidence intervals and SHAP-based feature importance analysis.

Functions:
    - bootstrap_evaluation: Performs bootstrapping to estimate confidence intervals for metrics.
    - traintest_finalmodel: Trains the final model and evaluates it on the test set.

Author: Julia Jelgerhuis
Last changed: 20 March 2025
"""

def bootstrap_evaluation(y_true, y_prob_pred, y_pred, n_iterations=1000):
    """
    Performs bootstrapping to estimate confidence intervals for classification metrics.
    
    Args:
        y_true (array-like): True labels.
        y_prob_pred (array-like): Predicted probabilities.
        y_pred (array-like): Predicted class labels.
        n_iterations (int): Number of bootstrap samples.
    
    Returns:
        dict: Bootstrapped confidence intervals for ROC AUC, specificity, and recall.
    """
    np.random.seed(50)
    metrics = {'roc_auc': [], 'specificity': [], 'recall': []}
    
    for _ in range(n_iterations):
        indices = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
        y_true_resampled = y_true[indices]
        y_prob_pred_resampled = y_prob_pred[indices]
        y_pred_resampled = y_pred[indices]
        
        fpr, tpr, _, roc_auc, _, recall, _, _, _, specificity = \
            calc_scoring(y_true_resampled, y_prob_pred_resampled, y_pred_resampled)
        
        metrics['roc_auc'].append(roc_auc)
        metrics['specificity'].append(specificity)
        metrics['recall'].append(recall)
    
    return {
        'roc_auc_ci': np.percentile(metrics['roc_auc'], [2.5, 97.5]),
        'specificity_ci': np.percentile(metrics['specificity'], [2.5, 97.5]),
        'recall_ci': np.percentile(metrics['recall'], [2.5, 97.5])
    }

def traintest_finalmodel(X_train, y_train, X_test, y_test, final_model, target, feature_set, n_bootstrap=1000):
    """
    Trains the final selected model on the full training set and evaluates it on the test set.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        final_model: Trained machine learning model.
        target (str): Target variable.
        feature_set (str): Selected feature set.
        n_bootstrap (int): Number of bootstrap iterations for confidence intervals.
    
    Returns:
        tuple: Final evaluation results, TPRs, Precision-Recall metrics, SHAP values, final trained model.
    """
    np.random.seed(50)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    y_prob_pred = final_model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, tpr_array, roc_auc, precision, recall, pr_auc, prec_array, f1, specificity = \
        calc_scoring(y_test, y_prob_pred, y_pred)
    
    metrics_ci = bootstrap_evaluation(y_test, y_prob_pred, y_pred, n_iterations=n_bootstrap)
    
    print(f'\nFinal Model Evaluation:')
    print(f'Median ROC AUC: {np.median(metrics_ci["roc_auc_ci"])} (95% CI: {metrics_ci["roc_auc_ci"]})')
    print(f'Median Recall: {np.median(metrics_ci["recall_ci"])} (95% CI: {metrics_ci["recall_ci"]})')
    print(f'Median Specificity: {np.median(metrics_ci["specificity_ci"])} (95% CI: {metrics_ci["specificity_ci"]})')
    
    # SHAP Analysis
    explainer = shap.Explainer(final_model.predict, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values.values, X_test, show=False)
    plt.title(f'SHAP Summary ({target}, {feature_set})')
    plt.savefig(f'./visualization/shap_summary_{target}_{feature_set}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return [final_model.__class__.__name__, roc_auc, precision, recall, pr_auc, f1, specificity], tpr_array, prec_array, shap_values, final_model
