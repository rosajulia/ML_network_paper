from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.pipeline import Pipeline
import numpy as np

"""
preprocessing.py - Data Preprocessing for Cognitive Impairment Classification

This module defines preprocessing steps for handling numerical, categorical, and ordinal data.
It includes scaling, encoding, and imputation for missing values.

Functions:
    - create_preprocessor: Constructs preprocessing pipelines based on selected feature sets.

Author: Julia Jelgerhuis
Last changed: 20 March 2025
"""

def create_preprocessor(feature_set):
    """
    Constructs preprocessing pipelines for numerical, categorical, and ordinal features.
    
    Args:
        feature_set (str): Selected feature set.
    
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    # Define feature groups
    feature_groups = {
        "Base": {"cat": ["Sex", "Edu"], "num": ["Age"]},
        "Base+Network": {"cat": ["Sex", "Edu"], "num": ["Age", "fmri_glob_STR", "fmri_DMN_STR", "fmri_FPN_STR", 
                                                          "fmri_DAN_STR", "fmri_VAN_STR", "fmri_VIS_STR", "fmri_SMN_STR", 
                                                          "fmri_DGM_STR", "fmri_DMN_ECM", "fmri_FPN_ECM", "fmri_DAN_ECM", 
                                                          "fmri_VAN_ECM", "fmri_VIS_ECM", "fmri_SMN_ECM", "fmri_DGM_ECM", 
                                                          "fmri_DMN_PART", "fmri_FPN_PART", "fmri_DAN_PART", "fmri_VAN_PART", 
                                                          "fmri_VIS_PART", "fmri_SMN_PART", "fmri_DGM_PART", "fmri_Eloc", 
                                                          "fmri_Eglob", "glob_STR", "DMN_STR", "FPN_STR", "DAN_STR", "VAN_STR", 
                                                          "VIS_STR", "SMN_STR", "DGM_STR", "DMN_ECM", "FPN_ECM", "DAN_ECM", 
                                                          "VAN_ECM", "VIS_ECM", "SMN_ECM", "DGM_ECM", "DMN_PART", "FPN_PART", 
                                                          "DAN_PART", "VAN_PART", "VIS_PART", "SMN_PART", "DGM_PART", "Eloc", "Eglob"]},
        "Base+Network+MRI": {"cat": ["Sex", "Edu"], "num": ["Age", "Norm_DGMVol", "Norm_CortexVol", "WM_FA", "LesionVol"]},
        "Base+Network+Clin": {"cat": ["Sex", "Edu", "MS_type", "Treatment"], "num": ["Age", "Symp_dur"], "ord": ["EDSS"]}
    }
    
    # Extract feature lists
    cat_cols = feature_groups.get(feature_set, {}).get("cat", [])
    num_cols = feature_groups.get(feature_set, {}).get("num", [])
    ord_cols = feature_groups.get(feature_set, {}).get("ord", [])
    
    # Define preprocessing steps
    numeric_transformer = Pipeline([
        ('imputer', IterativeImputer()),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    ordinal_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', IterativeImputer())
    ])
    
    # Assemble preprocessing pipeline
    transformers = []
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))
    if num_cols:
        transformers.append(("num", numeric_transformer, num_cols))
    if ord_cols:
        transformers.append(("ord", ordinal_transformer, ord_cols))
    
    preprocessor = ColumnTransformer(transformers)
    
    return preprocessor
