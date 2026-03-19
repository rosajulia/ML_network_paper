from scripts.evaluation import calc_scoring
import shap
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.utils import resample

# Bootstrapping function
def bootstrap_evaluation(y_true, y_prob_pred, y_pred, n_iterations=1000):
    np.random.seed(50)

    metrics = {'roc_auc': [], 'specificity': [], 'recall': []}
    for i in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
        y_true_resampled = y_true[indices]
        y_prob_pred_resampled = y_prob_pred[indices]
        y_pred_resampled = y_pred[indices]
        
        # Calculate metrics
        fpr, tpr, _, roc_auc, _, recall, _, _, _, specificity = \
            calc_scoring(y_true_resampled, y_prob_pred_resampled, y_pred_resampled)
        
        metrics['roc_auc'].append(roc_auc)
        metrics['specificity'].append(specificity)
        metrics['recall'].append(recall)
    
    return metrics

def traintest_finalmodel(X_train, y_train, X_test, y_test, final_model, target, feature_set, n_bootstrap=1000):
    np.random.seed(50)
    final_results = []
    model_name = final_model[2].__class__.__name__
    tprs = []
    precs = []

    # Train final model on whole training set
    final_model.fit(X_train, y_train)

    # Evaluate the final model on the test set
    y_pred = final_model.predict(X_test)
    y_prob_pred = final_model.predict_proba(X_test)[:, 1]

    fpr, tpr, tpr_array, roc_auc, precision, recall, pr_auc, prec_array, f1, specificity = \
        calc_scoring(y_test, y_prob_pred, y_pred)

    # Bootstrapping
    metrics = bootstrap_evaluation(y_test, y_prob_pred, y_pred, n_iterations=n_bootstrap)
    roc_auc_ci = np.percentile(metrics['roc_auc'], [2.5, 97.5])
    specificity_ci = np.percentile(metrics['specificity'], [2.5, 97.5])
    recall_ci = np.percentile(metrics['recall'], [2.5, 97.5])

    final_results.append([model_name, float(roc_auc), float(precision), float(recall), float(pr_auc), float(f1),
                          float(specificity), roc_auc_ci, specificity_ci, recall_ci])
    tprs.append([[model_name], [tpr_array]])
    precs.append([[model_name], [prec_array]])

    roc_auc_median = np.median(metrics['roc_auc'])
    specificity_median = np.median(metrics['specificity'])
    recall_median = np.median(metrics['recall'])

    print('\n median AUC:', roc_auc_median, '95% CI:', roc_auc_ci)
    print('\n precision:', precision)
    print('\n median recall:', recall_median, '95% CI:', recall_ci)
    print('\n f1:', f1)
    print('\n median specificity:', specificity_median, '95% CI:', specificity_ci)

    # Use SHAP to explain predictions
    explainer = shap.Explainer(final_model.predict, X_train)
    shap_values = explainer(X_test)

    # Create figures
    # ROC AUC PLOT
    plt.plot([0, 1], [0, 1], linestyle='--', label='No skill', lw=2, color='black')
    plt.plot(fpr, tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f)' % roc_auc, lw=2, alpha=0.9)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for {}'.format(model_name))
    plt.legend(loc='lower right')
    plt.savefig('./visualization/rocaucplot_final_model_{}_{}_{}_{}.png'
                .format(model_name, target, feature_set, str(datetime.now().replace(second=0, microsecond=0))))
    plt.close()

    # Assuming shap_values.values is a 2D array and features is a list of feature names


    ##### WARNING: ALWAYS LOOK AT THE ORDER OF THE COLUMNS IN X_TEST, ORDER MATTERS FOR SHAP


    features = list(X_test.columns)

    mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)

    # Sum of absolute Shapley values
    shap_sum = mean_abs_shap_values.sum()

    # Normalize each Shapley value
    normalized_shap_values = (mean_abs_shap_values / shap_sum) * 100

    # Round the normalized values to two decimal places
    normalized_shap_values_rounded = np.round(normalized_shap_values, 2)

    # Create a list of feature names with their normalized Shapley values
    feature_names = [f"{feature}: {shap_value}%" for feature, shap_value in zip(features, normalized_shap_values_rounded)]

    from matplotlib.colors import LinearSegmentedColormap

    # Generate colormap through matplotlib
    newCmap = LinearSegmentedColormap.from_list("", ['#006FF0','#F08100'])
    fpath = Path(mpl.get_data_path(), "Roboto-Regular.ttf")
    fpath_bold = Path(mpl.get_data_path(), "Roboto-Bold.ttf")
        

    shap.summary_plot(np.array(shap_values.values), X_test, feature_names=feature_names, show=False, color_bar=True,
                    max_display=10, cmap=newCmap)

    # Get the current figure and axes objects. from @GarrettCGraham code
    fig, ax = plt.gcf(), plt.gca()

    _, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(8, h*1.5)
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

    ax.set_title('Feature Importance for SDMT Classification Using Random Forest\nwith Base on Hold-Out Test Set', pad=15, font=fpath_bold, fontsize=16)
    ax.set_ylabel("Features with percentage\n normalized SHAP value", rotation=0, font=fpath_bold, fontsize=12)
    ax.yaxis.set_label_coords(-0.25 , 0.95)

    # Modifying color bar parameters
    plt.savefig('./visualization/normalized_shap_values_holdout_testset_{}_{}_{}_{}.png'
                .format(model_name, target, feature_set, datetime.now().replace(second=0, microsecond=0).strftime("%Y-%m-%d_%H:%M")),
                bbox_inches='tight', dpi=300)
    plt.close()

    return final_results, tprs, precs, shap_values, final_model
