import pandas as pd
from sklearn.model_selection import train_test_split

"""
data_loader.py - Data Loading and Splitting for Cognitive Impairment Classification

This module handles loading and preprocessing of datasets for cognitive classification in MS patients.
It includes feature encoding and dataset splitting.

Functions:
    - load_data: Loads and preprocesses dataset.
    - split_data: Splits the dataset into training and testing sets based on selected feature sets.

Author: Julia Jelgerhuis
Last changed: 20 March 2025
"""

def load_data(datafile1, datafile2, target):
    """
    Loads and preprocesses the dataset, merging clinical and connectivity data.

    Args:
        datafile1 (str): Path to clinical data file.
        datafile2 (str): Path to connectivity data file.
        target (str): Target variable for classification.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    dfclin = pd.read_excel(datafile1, index_col="ID")
    dfcon = pd.read_excel(datafile2, index_col="ID")
    df = dfclin.join(dfcon)
    
    # Encode categorical features
    categorical_columns = {
        'Sex': ["Female", "Male"],
        'Edu': ["High education", "Low education"],
        'MS_type': ["RRMS", "SPMS", "PPMS"],
        'Treatment': ["Yes", "No"],
        'SDMT_2SD': ["High SDMT", "Low SDMT"]
    }
    
    for col, categories in categorical_columns.items():
        df[col] = df[col].astype('category').cat.set_categories(categories).cat.codes
    
    if target == "CI-CP":
        df['CI-CP'] = df['CI-CP'].astype('category').cat.set_categories(["CP", "CI"]).cat.codes
    
    return df

def split_data(df, target, feature_set):
    """
    Splits the dataset into features (X) and target variable (y), and applies feature selection.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        target (str): Target variable.
        feature_set (str): Selected feature set.

    Returns:
        tuple: X_train, X_test, y_train, y_test, selected features, target variable.
    """
    feature_sets = {
        "Base": ["Sex", "Age", "Edu"],
        "Base+Network": ["Sex", "Age", "Edu", 'fmri_glob_STR', 'fmri_DMN_STR', 'fmri_FPN_STR',
                           'fmri_DAN_STR', 'fmri_VAN_STR', 'fmri_VIS_STR', 'fmri_SMN_STR', 'fmri_DGM_STR',
                           'fmri_DMN_ECM', 'fmri_FPN_ECM', 'fmri_DAN_ECM', 'fmri_VAN_ECM', 'fmri_VIS_ECM',
                           'fmri_SMN_ECM', 'fmri_DGM_ECM', 'fmri_DMN_PART', 'fmri_FPN_PART', 'fmri_DAN_PART',
                           'fmri_VAN_PART', 'fmri_VIS_PART', 'fmri_SMN_PART', 'fmri_DGM_PART', 'fmri_Eloc',
                           'fmri_Eglob', 'glob_STR', 'DMN_STR', 'FPN_STR', 'DAN_STR', 'VAN_STR', 'VIS_STR',
                           'SMN_STR', 'DGM_STR', 'DMN_ECM', 'FPN_ECM', 'DAN_ECM', 'VAN_ECM', 'VIS_ECM',
                           'SMN_ECM', 'DGM_ECM', 'DMN_PART', 'FPN_PART', 'DAN_PART', 'VAN_PART', 'VIS_PART',
                           'SMN_PART', 'DGM_PART', 'Eloc', 'Eglob'],
        "Base+Network+MRI": ["Sex", "Age", "Edu", "Norm_DGMVol", "Norm_CortexVol", "WM_FA", "LesionVol"],
        "Base+Network+Clin": ["Sex", "Age", "Edu", "EDSS", "Symp_dur", "MS_type", "Treatment"]
    }
    
    features = feature_sets.get(feature_set, [])
    if target in features:
        features.remove(target)
    
    df_filtered = df[df[target] != -1]
    X = df_filtered[features]
    y = df_filtered[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    
    return X_train, X_test, y_train, y_test, features, target
