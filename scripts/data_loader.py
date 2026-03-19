import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(datafile1, datafile2, target):
    """
    Load and preprocess your data, returning it as a DataFrame.

    Args:
        datafile1 (str): Path to the dataset file.
        datafile2 (str): Path to the dataset file.
        target (str): Binary target.

    Returns:
        pd.DataFrame: Preprocessed DataFrame(s).
    """
    dfclin = pd.read_excel(datafile1)
    dfclin.set_index(["ID"], inplace=True)

    dfcon = pd.read_excel(datafile2)
    dfcon.set_index(["ID"], inplace=True)

    df = dfclin.join(dfcon)

    # Encode categorical features
    df['Sex'] = df['Sex'].astype('category').cat. \
        set_categories(["Female", "Male"]).cat.codes
    df['Edu'] = df['Edu'].astype('category').cat. \
        set_categories(["High education", "Low education"]).cat.codes
    df['MS_type'] = df['MS_type'].astype('category').cat. \
        set_categories(["RRMS", "SPMS", "PPMS"]).cat.codes
    df['Treatment'] = df['Treatment'].astype('category').cat. \
        set_categories(["Yes", "No"]).cat.codes

    df['SDMT_2SD'] = df['SDMT_2SD'].astype('category').cat. \
        set_categories(["High SDMT", "Low SDMT"]).cat.codes

    if target == "CI-CP":
        df['CI-CP'] = df['CI-CP'].astype('category').cat. \
            set_categories(["CP", "CI"]).cat.codes

    return df


def split_data(df, target, feature_set):
    """
    Split the dataset into features (X) and the target variable (y).

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        target (str): Name of the target column.
        feature_set (str): Name of feature set used.

    Returns:
        pd.DataFrame, pd.Series: Features (X), Target (y).
    """

    features = list(df.columns)
    
    # Set feature-sets
    if feature_set == "Base":
        base_features = ["Sex", "Age", "Edu"] 
        features = [col for col in features if col in base_features]
        
    elif feature_set == "Base+Network":
        base_network_features = ["Sex", "Age", "Edu", 'fmri_glob_STR', 'fmri_DMN_STR', 'fmri_FPN_STR',
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
        features = [col for col in features if col in base_network_features]
        
    elif feature_set == "Base+Network+MRI":
        base_network_mri_features = ["Sex", "Age", "Edu", 'fmri_glob_STR', 'fmri_DMN_STR', 'fmri_FPN_STR',
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
        features = [col for col in features if col in base_network_mri_features]
        
    elif feature_set == "Base+Network+Clin":
        base_network_clin_features = ["Sex", "Age", "Edu", 'fmri_glob_STR', 'fmri_DMN_STR', 'fmri_FPN_STR',
                                    'fmri_DAN_STR', 'fmri_VAN_STR', 'fmri_VIS_STR', 'fmri_SMN_STR',
                                    'fmri_DGM_STR', 'fmri_DMN_ECM', 'fmri_FPN_ECM', 'fmri_DAN_ECM',
                                    'fmri_VAN_ECM', 'fmri_VIS_ECM', 'fmri_SMN_ECM', 'fmri_DGM_ECM',
                                    'fmri_DMN_PART', 'fmri_FPN_PART', 'fmri_DAN_PART', 'fmri_VAN_PART',
                                    'fmri_VIS_PART', 'fmri_SMN_PART', 'fmri_DGM_PART', 'fmri_Eloc',
                                    'fmri_Eglob', 'glob_STR', 'DMN_STR', 'FPN_STR', 'DAN_STR', 'VAN_STR',
                                    'VIS_STR', 'SMN_STR', 'DGM_STR', 'DMN_ECM', 'FPN_ECM', 'DAN_ECM',
                                    'VAN_ECM', 'VIS_ECM', 'SMN_ECM', 'DGM_ECM', 'DMN_PART', 'FPN_PART',
                                    'DAN_PART', 'VAN_PART', 'VIS_PART', 'SMN_PART', 'DGM_PART', 'Eloc',
                                    'Eglob',"EDSS", "Symp_dur", "MS_type", "Treatment"]
        
        features = [col for col in features if col in base_network_clin_features]
    
    elif feature_set == "Base+FuncNetwork":
        base_func_features = ["Sex", "Age", "Edu", 'fmri_glob_STR', 'fmri_DMN_STR', 'fmri_FPN_STR',
                                    'fmri_DAN_STR', 'fmri_VAN_STR', 'fmri_VIS_STR', 'fmri_SMN_STR',
                                    'fmri_DGM_STR', 'fmri_DMN_ECM', 'fmri_FPN_ECM', 'fmri_DAN_ECM',
                                    'fmri_VAN_ECM', 'fmri_VIS_ECM', 'fmri_SMN_ECM', 'fmri_DGM_ECM',
                                    'fmri_DMN_PART', 'fmri_FPN_PART', 'fmri_DAN_PART', 'fmri_VAN_PART',
                                    'fmri_VIS_PART', 'fmri_SMN_PART', 'fmri_DGM_PART', 'fmri_Eloc',
                                    'fmri_Eglob']
        
        features = [col for col in features if col in base_func_features]

    elif feature_set == "Base+StructNetwork":
        base_struct_features = ["Sex", "Age", "Edu", 'glob_STR', 'DMN_STR', 'FPN_STR', 'DAN_STR', 'VAN_STR',
                                    'VIS_STR', 'SMN_STR', 'DGM_STR', 'DMN_ECM', 'FPN_ECM', 'DAN_ECM',
                                    'VAN_ECM', 'VIS_ECM', 'SMN_ECM', 'DGM_ECM', 'DMN_PART', 'FPN_PART',
                                    'DAN_PART', 'VAN_PART', 'VIS_PART', 'SMN_PART', 'DGM_PART', 'Eloc',
                                    'Eglob']
        
        features = [col for col in features if col in base_struct_features]

    elif feature_set == "Base+Clin":
        base_clin_features = ["Sex", "Age", "Edu", "EDSS", "Symp_dur", "MS_type", "Treatment"]
        
        features = [col for col in features if col in base_clin_features]

    elif feature_set == "Base+MRI":
        base_mri_features = ["Sex", "Age", "Edu", "Norm_DGMVol", "Norm_CortexVol", "WM_FA", "LesionVol"]
        
        features = [col for col in features if col in base_mri_features]

    # remove target from feature list
    if target in features:
        features.remove(target)

    # remove features closely related to target
    if target == "CI-CP":
        df_filtered = df[df['CI-CP'] != -1]

    elif target == "SDMT_2SD":
        df_filtered = df[df['SDMT_2SD'] != -1]

    # Split train and test data
    X = df_filtered[features]
    y = df_filtered[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test, features, target
