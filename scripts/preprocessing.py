from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.pipeline import Pipeline
import numpy as np


def create_preprocessor(feature_set):
    """
    Create a preprocessing pipeline for your data.

    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """

    if feature_set == "Base":
        cat_cols = ["Sex", "Edu"]
        num_cols = ['Age']

    elif feature_set == "Base+Network":
        cat_cols = ["Sex", "Edu"]
        num_cols = ['Age', 
                    'fmri_glob_STR', 'fmri_DMN_STR', 'fmri_FPN_STR',
                    'fmri_DAN_STR', 'fmri_VAN_STR', 'fmri_VIS_STR', 'fmri_SMN_STR',
                    'fmri_DGM_STR', 'fmri_DMN_ECM', 'fmri_FPN_ECM', 'fmri_DAN_ECM',
                    'fmri_VAN_ECM', 'fmri_VIS_ECM', 'fmri_SMN_ECM', 'fmri_DGM_ECM',
                    'fmri_DMN_PART', 'fmri_FPN_PART', 'fmri_DAN_PART', 'fmri_VAN_PART',
                    'fmri_VIS_PART', 'fmri_SMN_PART', 'fmri_DGM_PART', 'fmri_Eloc',
                    'fmri_Eglob', 'glob_STR', 'DMN_STR', 'FPN_STR', 'DAN_STR', 'VAN_STR',
                    'VIS_STR', 'SMN_STR', 'DGM_STR', 'DMN_ECM', 'FPN_ECM', 'DAN_ECM',
                    'VAN_ECM', 'VIS_ECM', 'SMN_ECM', 'DGM_ECM', 'DMN_PART', 'FPN_PART',
                    'DAN_PART', 'VAN_PART', 'VIS_PART', 'SMN_PART', 'DGM_PART', 'Eloc',
                    'Eglob']

    elif feature_set == "Base+Network+MRI":
        cat_cols = ["Sex", "Edu"]
        num_cols = ['Age',
                    'fmri_glob_STR', 'fmri_DMN_STR', 'fmri_FPN_STR',
                    'fmri_DAN_STR', 'fmri_VAN_STR', 'fmri_VIS_STR', 'fmri_SMN_STR',
                    'fmri_DGM_STR', 'fmri_DMN_ECM', 'fmri_FPN_ECM', 'fmri_DAN_ECM',
                    'fmri_VAN_ECM', 'fmri_VIS_ECM', 'fmri_SMN_ECM', 'fmri_DGM_ECM',
                    'fmri_DMN_PART', 'fmri_FPN_PART', 'fmri_DAN_PART', 'fmri_VAN_PART',
                    'fmri_VIS_PART', 'fmri_SMN_PART', 'fmri_DGM_PART', 'fmri_Eloc',
                    'fmri_Eglob', 'glob_STR', 'DMN_STR', 'FPN_STR', 'DAN_STR', 'VAN_STR',
                    'VIS_STR', 'SMN_STR', 'DGM_STR', 'DMN_ECM', 'FPN_ECM', 'DAN_ECM',
                    'VAN_ECM', 'VIS_ECM', 'SMN_ECM', 'DGM_ECM', 'DMN_PART', 'FPN_PART',
                    'DAN_PART', 'VAN_PART', 'VIS_PART', 'SMN_PART', 'DGM_PART', 'Eloc',
                    'Eglob', "Norm_DGMVol", "Norm_CortexVol", "WM_FA", "LesionVol"]
        
    elif feature_set == "Base+Network+Clin":
        cat_cols = ["Sex", "Edu", "MS_type", "Treatment"]
        num_cols = ['Age', 'Symp_dur', 'fmri_glob_STR', 'fmri_DMN_STR', 'fmri_FPN_STR',
        'fmri_DAN_STR', 'fmri_VAN_STR', 'fmri_VIS_STR', 'fmri_SMN_STR',
        'fmri_DGM_STR', 'fmri_DMN_ECM', 'fmri_FPN_ECM', 'fmri_DAN_ECM',
        'fmri_VAN_ECM', 'fmri_VIS_ECM', 'fmri_SMN_ECM', 'fmri_DGM_ECM',
        'fmri_DMN_PART', 'fmri_FPN_PART', 'fmri_DAN_PART', 'fmri_VAN_PART',
        'fmri_VIS_PART', 'fmri_SMN_PART', 'fmri_DGM_PART', 'fmri_Eloc',
        'fmri_Eglob', 'glob_STR', 'DMN_STR', 'FPN_STR', 'DAN_STR', 'VAN_STR',
        'VIS_STR', 'SMN_STR', 'DGM_STR', 'DMN_ECM', 'FPN_ECM', 'DAN_ECM',
        'VAN_ECM', 'VIS_ECM', 'SMN_ECM', 'DGM_ECM', 'DMN_PART', 'FPN_PART',
        'DAN_PART', 'VAN_PART', 'VIS_PART', 'SMN_PART', 'DGM_PART', 'Eloc',
        'Eglob']
        ord_cols = ["EDSS"]
    
    elif feature_set == "Base+FuncNetwork":
        cat_cols = ["Sex", "Edu"]
        num_cols = ['Age',
                    'fmri_glob_STR', 'fmri_DMN_STR', 'fmri_FPN_STR',
                    'fmri_DAN_STR', 'fmri_VAN_STR', 'fmri_VIS_STR', 'fmri_SMN_STR',
                    'fmri_DGM_STR', 'fmri_DMN_ECM', 'fmri_FPN_ECM', 'fmri_DAN_ECM',
                    'fmri_VAN_ECM', 'fmri_VIS_ECM', 'fmri_SMN_ECM', 'fmri_DGM_ECM',
                    'fmri_DMN_PART', 'fmri_FPN_PART', 'fmri_DAN_PART', 'fmri_VAN_PART',
                    'fmri_VIS_PART', 'fmri_SMN_PART', 'fmri_DGM_PART', 'fmri_Eloc',
                    'fmri_Eglob']
        
    elif feature_set == "Base+StructNetwork":
        cat_cols = ["Sex", "Edu"]
        num_cols = ['Age','glob_STR', 'DMN_STR', 'FPN_STR', 'DAN_STR', 'VAN_STR',
                    'VIS_STR', 'SMN_STR', 'DGM_STR', 'DMN_ECM', 'FPN_ECM', 'DAN_ECM',
                    'VAN_ECM', 'VIS_ECM', 'SMN_ECM', 'DGM_ECM', 'DMN_PART', 'FPN_PART',
                    'DAN_PART', 'VAN_PART', 'VIS_PART', 'SMN_PART', 'DGM_PART', 'Eloc',
                    'Eglob']
        
    elif feature_set == "Base+Clin":
        cat_cols = ["Sex", "Edu", "MS_type", "Treatment"]
        num_cols = ['Age', 'Symp_dur']
        ord_cols = ["EDSS"]

    elif feature_set == "Base+MRI":
        cat_cols = ["Sex", "Edu"]
        num_cols = ['Age', "Norm_DGMVol", "Norm_CortexVol", "WM_FA", "LesionVol"]  
        
    # create transformers for different features types
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', IterativeImputer()),
            ('num_scaler', StandardScaler())
        ])

    categorical_transformer = Pipeline(
        steps=[
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore',
                                         categories='auto'))
        ])

    ordinal_transformer = Pipeline(
        steps=[
            ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',
                                           unknown_value=np.nan)),
            ('imputer', IterativeImputer())
        ])

    # merge transformers into preprocessor column
    
    if feature_set == "Base+Network+Clin" or feature_set == "Base+Clin":
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, cat_cols),
                ("num", numeric_transformer, num_cols),
                ("ord", ordinal_transformer, ord_cols)
            ])
    
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, cat_cols),
                ("num", numeric_transformer, num_cols)
            ])
        

    return preprocessor
