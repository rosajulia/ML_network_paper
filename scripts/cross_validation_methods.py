from sklearn.model_selection import GridSearchCV, StratifiedKFold
import shap
import numpy as np
from scripts.evaluation import calc_scoring
import signal

def fit_with_timeout(gcv_model_select, X_train, y_train, timeout):
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
#def nestedCV(X, y, n_folds_inner, n_folds_outer, pipeline, hyperparams, scoring, names, timeout=600):
 
    """
    Perform nested cross-validation on data with a timeout mechanism.
    """
    outer_fold_results = []
    tprs = []
    precs = []
    
    # Initialize dictionaries for storing SHAP values
    shap_values_per_cv = {sample: {CV_repeat: {} for CV_repeat in range(repetitions)} for sample in X.index}

    # start repetition loop
    for CV_repeat in range(repetitions):
        print('\n------------ CV Repeat number:', CV_repeat)

        # set stratified K fold for both inner and outer loop
        skf_outer = StratifiedKFold(n_splits=n_folds_outer, shuffle=True, random_state=CV_repeat)
        skf_inner = StratifiedKFold(n_splits=n_folds_inner, shuffle=True, random_state=CV_repeat)

        # start outer loop
        for i, (train_index, test_index) in enumerate(skf_outer.split(X, y)):
            print('\n------ Fold Number:', i)

            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # start inner loop (tuning)
            for est, pgrid, name in zip(pipeline.values(), hyperparams, names):
                print('\n------ Algorithm:', name)
                gcv_model_select = GridSearchCV(estimator=est,
                                                param_grid=pgrid,
                                                scoring=scoring,
                                                n_jobs=8,  
                                                cv=skf_inner,
                                                verbose=1,
                                                refit="AUC")

                # fit grid search object on training data with timeout
                best_model, best_params = fit_with_timeout(gcv_model_select, X_train, y_train, timeout)
                
                if best_model is None:
                    print(f"Timeout reached for fold {i} with algorithm {name}, skipping this fold.")
                    # Append default values for metrics
                    outer_fold_results.append([int(CV_repeat), int(i), name,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None])
                    tprs.append([[name], [np.zeros_like(y_test)]])
                    precs.append([[name], [np.zeros_like(y_test)]])
                    continue
                
                print('\n Best parameters:', best_params)

                prob_pred = best_model.predict_proba(X_test)[:, 1]
                pred = best_model.predict(X_test)

                fpr, tpr, tpr_array, roc_auc, precision, recall, pr_auc, prec_array, f1, specificity = \
                    calc_scoring(y_test, prob_pred, pred)

                outer_fold_results.append([int(CV_repeat), int(i), name,
                                           float(roc_auc), float(precision), float(recall), float(pr_auc),
                                           float(f1), float(specificity), best_params])
                tprs.append([[name], [tpr_array]])
                precs.append([[name], [prec_array]])

                print('\n AUC:', roc_auc)
                print('\n precision:', precision)
                print('\n recall:', recall)
                print('\n f1:', f1)
                print('\n specificity:', specificity)

                # Use SHAP to explain predictions
                explainer = shap.Explainer(best_model.predict, X_train)
                shap_values = explainer(X_test)

                # Extract SHAP information per fold per sample
                for j, test_ix in enumerate(test_index):
                    shap_alg = {name: shap_values[j]}
                    shap_values_per_cv[X.index[test_ix]][CV_repeat].update(shap_alg)

                # Explicitly delete explainer to free memory
                del explainer, shap_values

    return outer_fold_results, tprs, precs, shap_values_per_cv
