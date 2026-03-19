import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, precision_recall_curve
from sklearn.metrics import recall_score
import matplotlib as mpl

import csv

import pickle
from pathlib import Path

from datetime import datetime


def calc_scoring(y_true, prob_pred, pred):
    """
    Calculate scoring metrics (AUC, precision, recall, F1, specificity).

    Args:
        y_true (pd.Series): True labels.
        prob_pred (array-like): Predicted probabilities.
        pred (array-like): Predicted labels.

    Returns:
        tuple: fpr, tpr, AUC, precision, recall, F1, specificity.
    """

    # calculate scoring metrics
    thresholds = np.linspace(0, 1, 100)

    fpr, tpr, _ = roc_curve(y_true, prob_pred)
    roc_auc = auc(fpr, tpr)
    tpr_array = np.interp(thresholds, fpr, tpr)

    precision_fold, recall_fold, _ = precision_recall_curve(y_true, prob_pred)
    precision_fold, recall_fold = precision_fold[::-1], recall_fold[::-1]  # reverse order of results
    prec_array = np.interp(thresholds, recall_fold, precision_fold)

    pr_auc = auc(thresholds, prec_array)

    precision = precision_score(y_true, pred).ravel()
    recall = recall_score(y_true, pred).ravel()

    f1 = 2 * (precision * recall) / (precision + recall)

    specificity = recall_score(y_true, pred, pos_label=0).ravel()

    return fpr, tpr, tpr_array, roc_auc, precision, recall, pr_auc, prec_array, f1, specificity

def create_shap_plots(X, shap_dict, name, target, feature_set):
    """
    Create SHAP summary plots.

    Args:
        X (pd.DataFrame): Training data.
        shap_dict (dict): Dictionary of SHAP values.
        name (str): Model name.
        target (str): Name of target.
        feature_set (list): List of features.

    Returns:
        None
    """
# Initialize a dictionary to store aggregated SHAP values
    aggregated_shap_values = {}

    for subj, data in shap_dict.items():
        # Initialize a list to collect SHAP values for all CV iterations
        shap_values_all_iterations = []

        for iteration in data.keys():
            # Extract the SHAP values array for the current iteration
            shap_values = data[iteration].values
            shap_values_all_iterations.append(shap_values)

        # Convert elements to numpy arrays if not already done
        shap_values_all_iterations = [np.array(item) for item in shap_values_all_iterations]

        # Find the maximum length of the arrays
        max_length = max(len(item) for item in shap_values_all_iterations)

        # Pad arrays to the maximum length
        padded_shap_values = []
        for item in shap_values_all_iterations:
            if len(item) < max_length:
                padded_item = np.pad(item, (0, max_length - len(item)), 'constant', constant_values=(0,))
            else:
                padded_item = item
            padded_shap_values.append(padded_item)

        # Convert to numpy array
        padded_shap_values = np.array(padded_shap_values)

        # Calculate the mean SHAP values across all iterations for the current sample
        mean_shap_values = np.mean(padded_shap_values, axis=0)

        # Store the aggregated SHAP values in the dictionary
        aggregated_shap_values[subj] = mean_shap_values

    # Ensure all SHAP values have the same shape by padding them to the maximum length
    max_length_final = max(len(v) for v in aggregated_shap_values.values())
    for subj in aggregated_shap_values:
        if len(aggregated_shap_values[subj]) < max_length_final:
            aggregated_shap_values[subj] = np.pad(aggregated_shap_values[subj], 
                                                  (0, max_length_final - len(aggregated_shap_values[subj])), 
                                                  'constant', constant_values=(0,))

    # Extract SHAP values in the order of `X`'s index
    ordered_shap_values = np.array([aggregated_shap_values[subjID] for subjID in X.index])


    from matplotlib.colors import LinearSegmentedColormap

    # Generate colormap through matplotlib
    newCmap = LinearSegmentedColormap.from_list("", ['#006FF0', '#F08100'])
    fpath = Path(mpl.get_data_path(), "/home/anw/jrjelgerhuis/python-env/fonts/Roboto/Roboto-Regular.ttf")
    fpath_bold = Path(mpl.get_data_path(), "/home/anw/jrjelgerhuis/python-env/fonts/Roboto/Roboto-Bold.ttf")
    
    shap_values = np.array([item[0].values for item in ordered_shap_values])
    shap.summary_plot(shap_values, X, show=False, color_bar=True, max_display=10, cmap=newCmap)

    # Get the current figure and axes objects. from @GarrettCGraham code
    fig, ax = plt.gcf(), plt.gca()

    _, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(8, h * 1.5)
    fig.subplots_adjust(
        top=0.94,
        bottom=0.181,
        left=0.213,
        right=0.981,
        hspace=0.2,
        wspace=0.2)

    # Modifying main plot parameters
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(fpath)
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(fpath)
        tick.set_fontsize(12)

    ax.set_title('Average SHAP values after nested CV \n'
              'Algorithm: {}, target: {}, feature set: {}'
              .format(name, target, feature_set), pad=15, font=fpath_bold, fontsize=16)
    #ax.set_ylabel("Features with percentage\n normalized SHAP value", rotation=0, font=fpath_bold, fontsize=12)
    ax.yaxis.set_label_coords(-0.25, 0.95)

    # Modifying color bar parameters
    plt.savefig('./visualization/shapplot_{}_{}_{}_{}.png'
                .format(name, target, feature_set, str(datetime.now().replace(second=0, microsecond=0))),
                bbox_inches='tight', dpi=300)
    plt.close()

def prec_recall_plot(precision, name, target, feature_set):
    thresholds = np.linspace(0, 1, 100)

    # Find the maximum length of the sequences
    max_length = max(len(p) for p in precision)

    # Pad the sequences to the maximum length
    padded_precision = []
    for p in precision:
        if len(p) < max_length:
            padded_p = np.pad(p, (0, max_length - len(p)), 'constant', constant_values=(0,))
        else:
            padded_p = p
        padded_precision.append(padded_p)

    # Convert to numpy array
    padded_precision = np.array(padded_precision)

    # Calculate the mean precision
    mean_prec = np.mean(padded_precision, axis=0)

    mean_auc = auc(thresholds, mean_prec)

    std_prec = np.std(padded_precision, axis=0)

    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = mean_prec - std_prec
    plt.plot(thresholds, mean_prec, color='blue',
             label=r'Mean Precision (AUC = %0.2f)' % mean_auc, lw=2, alpha=0.9)
    plt.fill_between(thresholds, prec_lower, prec_upper, color='grey', alpha=0.3)

    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall curve \n'
              'Algorithm: {}, target: {}, feature set: {}'
              .format(name, target, feature_set))
    plt.legend(loc='lower right')
    plt.savefig('./visualization/prcurveplot_ROC_{}_{}_{}_{}.png'
                .format(name, target, feature_set, str(datetime.now().replace(second=0, microsecond=0))))
    plt.close()


def roc_auc_plot(tprs, name, target, feature_set):
    """
    Create ROC AUC plots.

    Args:
        tprs (list): List of true positive rates.
        name (str): Algorithm name.
        target (str): Name of target.

    Returns:
        None
    """
    thresholds = np.linspace(0, 1, 100)
    # If you want all the AUC lines in the plot
    # for i in range(len(tprs)):
    #   plt.plot(mean_fpr, tprs[i], alpha=0.15)

    plt.plot([0, 1], [0, 1], linestyle='--', label='No skill', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(thresholds, mean_tpr)

    std_tpr = np.std(tprs, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = mean_tpr - std_tpr
    plt.plot(thresholds, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f)' % mean_auc, lw=2, alpha=0.9)
    plt.fill_between(thresholds, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic \n'
              'Algorithm: {}, target: {}, feature set: {}'
              .format(name, target, feature_set))
    plt.legend(loc='lower right')
    plt.savefig('./visualization/rocaucplot_{}_{}_{}_{}.png'
                .format(name, target, feature_set, str(datetime.now().replace(second=0, microsecond=0))))
    plt.close()

def save_variables(results, tprs, shap_values, target, feature_set, best_alg, high_rocauc, opt_hp):
    with open('./visualization/perf_metrics_nestedcv_{}_{}_{}.csv'
                      .format(target, feature_set, str(datetime.now().replace(second=0, microsecond=0))),
              'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["Repeat", "Fold number", "Algorithm", "ROC AUC", "Precision", "Recall", "PR AUC" "F1", "Specificity"])
        csv_writer.writerows(results)

    with open('./visualization/best_model_parameters_{}_{}_{}.txt'
                      .format(target, feature_set, str(datetime.now().replace(second=0, microsecond=0))),
              'w') as t:
        # Redirect print statements to the file
        print(f"For the best algorithm {best_alg} trained on {feature_set}, "
              f"the set of hyperparameters with the highest ROC AUC "
              f"({high_rocauc}) is:", file=t)
        print(f"Parameter Set: {opt_hp}", file=t)

    # save dictionary to person_data.pkl file
    with open('./visualization/shapvalues_nestedcv_{}_{}_{}.pkl'
                      .format(target, feature_set, str(datetime.now().replace(second=0, microsecond=0))), 'wb') as fp:
        pickle.dump(shap_values, fp)

    # save shapley values and tprs as numpy arrays, load with np.load
    with open('./visualization/tprs_nestedcv_{}_{}_{}.pkl'.format(target, feature_set, str(datetime.now().replace(second=0, microsecond=0))), 'wb') as f:
        pickle.dump(tprs, f)


def save_variables_final_model(results, tprs, shap_values, target, feature_set, final_model):
    with open('./visualization/perf_metrics_final_model_{}_{}_{}.csv'
                      .format(target, feature_set, str(datetime.now().replace(second=0, microsecond=0))), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["Algorithm", "ROC AUC", "Precision", "Recall", "PR AUC" "F1", "Specificity"])
        csv_writer.writerows(results)

    # save dictionary to person_data.pkl file
    with open('./visualization/shapvalues_final_model_{}_{}_{}.pkl'
                      .format(target, feature_set, str(datetime.now().replace(second=0, microsecond=0))), 'wb') as fp:
        pickle.dump(shap_values, fp)

    # save shapley values and tprs as numpy arrays, load with np.load
    with open('./visualization/tprs_final_model_{}_{}_{}.npy'.format(target, feature_set, str(datetime.now().replace(second=0, microsecond=0))), 'wb') as f:
        pickle.dump(tprs, f)
    # save final model to .pkl file
    with open('./visualization/final_model_{}_{}_{}.pkl'
                      .format(target, feature_set, str(datetime.now().replace(second=0, microsecond=0))), 'wb') as fp:
        pickle.dump(final_model, fp)
