import shap
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_score, precision_recall_curve, recall_score

"""
evaluation.py - Model Evaluation and Visualization for Cognitive Impairment Classification

This module provides functions to compute performance metrics, generate plots (ROC-AUC, precision-recall),
and save results for trained models.

Functions:
    - calc_scoring: Computes classification metrics including AUC, precision, recall, and F1-score.
    - create_shap_plots: Generates SHAP summary plots for feature importance.
    - prec_recall_plot: Plots precision-recall curves.
    - roc_auc_plot: Plots ROC-AUC curves.
    - save_variables: Saves model performance metrics and SHAP values.
    - save_variables_final_model: Saves evaluation metrics for the final trained model.

Author: Julia Jelgerhuis
Last changed: 20 March 2025
"""

def calc_scoring(y_true, prob_pred, pred):
    """
    Calculate classification performance metrics.
    
    Args:
        y_true (array-like): True class labels.
        prob_pred (array-like): Predicted probabilities.
        pred (array-like): Predicted class labels.
    
    Returns:
        tuple: fpr, tpr, tpr_array, roc_auc, precision, recall, pr_auc, prec_array, f1, specificity
    """
    thresholds = np.linspace(0, 1, 100)
    fpr, tpr, _ = roc_curve(y_true, prob_pred)
    roc_auc = auc(fpr, tpr)
    tpr_array = np.interp(thresholds, fpr, tpr)

    precision_fold, recall_fold, _ = precision_recall_curve(y_true, prob_pred)
    precision_fold, recall_fold = precision_fold[::-1], recall_fold[::-1]  # Reverse order
    prec_array = np.interp(thresholds, recall_fold, precision_fold)
    pr_auc = auc(thresholds, prec_array)
    
    precision = precision_score(y_true, pred)
    recall = recall_score(y_true, pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    specificity = recall_score(y_true, pred, pos_label=0)
    
    return fpr, tpr, tpr_array, roc_auc, precision, recall, pr_auc, prec_array, f1, specificity

def create_shap_plots(X, shap_dict, name, target, feature_set):
    """
    Generates SHAP summary plots for feature importance.
    
    Args:
        X (pd.DataFrame): Training data.
        shap_dict (dict): Dictionary of SHAP values.
        name (str): Model name.
        target (str): Target variable.
        feature_set (str): Selected feature set.
    """
    shap_values = np.array([np.mean([shap_dict[sample][cv][name] for cv in shap_dict[sample] if name in shap_dict[sample][cv]], axis=0)
                            for sample in X.index])
    
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP Summary Plot ({name}, {target}, {feature_set})')
    plt.savefig(f'./visualization/shapplot_{name}_{target}_{feature_set}.png', bbox_inches='tight', dpi=300)
    plt.close()

def prec_recall_plot(precision, name, target, feature_set):
    """
    Generates and saves precision-recall curve plots.
    
    Args:
        precision (list): List of precision values.
        name (str): Model name.
        target (str): Target variable.
        feature_set (str): Selected feature set.
    """
    thresholds = np.linspace(0, 1, 100)
    mean_prec = np.mean(precision, axis=0)
    pr_auc = auc(thresholds, mean_prec)
    
    plt.plot(thresholds, mean_prec, label=f'Mean Precision (AUC = {pr_auc:.2f})', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({name}, {target}, {feature_set})')
    plt.legend()
    plt.savefig(f'./visualization/prcurve_{name}_{target}_{feature_set}.png', bbox_inches='tight', dpi=300)
    plt.close()

def roc_auc_plot(tprs, name, target, feature_set):
    """
    Generates and saves ROC-AUC curve plots.
    
    Args:
        tprs (list): List of true positive rates.
        name (str): Model name.
        target (str): Target variable.
        feature_set (str): Selected feature set.
    """
    thresholds = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    roc_auc = auc(thresholds, mean_tpr)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.plot(thresholds, mean_tpr, label=f'Mean ROC (AUC = {roc_auc:.2f})', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({name}, {target}, {feature_set})')
    plt.legend()
    plt.savefig(f'./visualization/roc_auc_{name}_{target}_{feature_set}.png', bbox_inches='tight', dpi=300)
    plt.close()

def save_variables(results, tprs, shap_values, target, feature_set, best_alg, high_rocauc, opt_hp):
    """
    Saves model performance metrics and SHAP values.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    with open(f'./visualization/performance_{target}_{feature_set}_{timestamp}.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Repeat", "Fold", "Algorithm", "ROC AUC", "Precision", "Recall", "PR AUC", "F1", "Specificity"])
        csv_writer.writerows(results)
    
    with open(f'./visualization/shap_values_{target}_{feature_set}_{timestamp}.pkl', 'wb') as fp:
        pickle.dump(shap_values, fp)
    
    with open(f'./visualization/tprs_{target}_{feature_set}_{timestamp}.pkl', 'wb') as f:
        pickle.dump(tprs, f)

def save_variables_final_model(results, tprs, shap_values, target, feature_set, final_model):
    """
    Saves evaluation metrics for the final trained model.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    with open(f'./visualization/final_model_{target}_{feature_set}_{timestamp}.pkl', 'wb') as fp:
        pickle.dump(final_model, fp)
    save_variables(results, tprs, shap_values, target, feature_set, "Final_Model", 0, {})
