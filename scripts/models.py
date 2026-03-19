from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline


def create_pipelines(preprocessor):
    """
    Create machine learning models.

    Returns:
        dict: Dictionary of models.
    """
    randfor = RandomForestClassifier(random_state=50)
    logreg = LogisticRegression(random_state=50)
    svm = SVC(probability=True, random_state=50)

    smt = SMOTETomek(random_state=50)

    pipe1 = Pipeline([('preprocessor', preprocessor),
                      ('smt', smt),
                      ('rf', randfor)])

    pipe2 = Pipeline([('preprocessor', preprocessor),
                      ('smt', smt),
                      ('lg', logreg)])

    pipe3 = Pipeline([('preprocessor', preprocessor),
                      ('smt', smt),
                      ('svm', svm)])

    pipelines = {
        'pipe_RF': pipe1,
        'pipe_LR': pipe2,
        'pipe_svm': pipe3
    }
    return pipelines


def set_hyperparams():
    hyperparams = {
        "RandomForest": {'rf__n_estimators': [50, 75, 100],
                         'rf__max_features': ["sqrt", "log2"],
                         'rf__max_depth': [5, 7, 9],
                         'rf__max_samples': [0.3, 0.5, 0.8],
                         'rf__criterion': ['entropy'],
                         'rf__class_weight': ["balanced_subsample"]},

        'LogisticRegression': [{'lg__penalty': ['l2'],
                                'lg__solver': ['lbfgs'],
                                'lg__max_iter': [1000],
                                'lg__C': [0.001, 0.01, 0.1, 1, 10, 100]},
                               {'lg__penalty': ['l1', 'l2'],
                                'lg__solver': ['liblinear'],
                                'lg__max_iter': [1000],
                                'lg__C': [0.001, 0.01, 0.1, 1, 10, 100]}
                               ],
        "SVM": {'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': [1, 0.1, 0.01, 0.001],
                'svm__kernel': ['rbf', 'poly', 'sigmoid']}

    }
    return hyperparams
