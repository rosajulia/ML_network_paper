# Cognitive Impairment Classification in Multiple Sclerosis

This repository contains Python scripts to classify cognitive impairment (CI) in Multiple Sclerosis (MS) using machine learning models trained on structural and functional brain network features. The project applies nested cross-validation, hyperparameter tuning, and model evaluation techniques, utilizing Shapley values for feature importance interpretation.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Feature Sets](#feature-sets)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview
Cognitive impairment affects a significant proportion of MS patients. This project leverages multimodal MRI data, including structural and functional connectivity measures, to classify cognitive impairment. The classification models are trained and evaluated on anonymized datasets.

Key findings include:
- Structural and functional brain network features contribute significantly to classification.
- Deep gray matter (DGM) and dorsal attention network (DAN) connectivity are the most relevant predictors.
- Adding MRI morphology features improves classification accuracy.

## Installation
### Requirements
Ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

### Required Packages
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- SHAP
- Matplotlib
- Scipy
- Openpyxl

## Usage
### Running the Main Script
To execute the classification pipeline, run:

```bash
python main.py -fs <feature_set> -t <target>
```

**Arguments:**
- `-fs`, `--feature_set`: Specifies the feature set. Options: `Base`, `Base+Network`, `Base+Network+MRI`, `Base+Network+Clin`.
- `-t`, `--target`: Specifies the target variable. Options: `SDMT_2SD` or `CI-CP`.

Example:
```bash
python main.py -fs Base+Network -t CI-CP
```

## Project Structure
```
├── scripts/                   # Python scripts
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── preprocessing.py       # Feature scaling and encoding
│   ├── models.py              # Model creation and hyperparameter tuning
│   ├── cross_validation_methods.py # Nested cross-validation
│   ├── train_test_final_model.py   # Final model training and testing
│   ├── evaluation.py          # Performance metrics and visualization
│
├── visualization/             # Output plots and results
│
├── main.py                    # Entry point script
└── README.md                   # Project documentation
```

## Feature Sets
Different feature sets are used for classification:
- **Base**: Age, Sex, Education.
- **Base+Network**: Base + connectivity metrics (functional and structural network measures).
- **Base+Network+MRI**: Base+Network + MRI-derived volumetric measures.
- **Base+Network+Clin**: Base+Network + clinical measures (MS subtype, disease duration, etc.).

## Model Training and Evaluation
1. **Data Preprocessing**:
   - Missing values are imputed.
   - Features are standardized.
   - Categorical variables are one-hot encoded.

2. **Model Selection**:
   - Logistic Regression, Random Forest, and Support Vector Machine (SVM) models are trained.
   - Hyperparameters are optimized using nested cross-validation.
   - Class imbalance is handled using SMOTETomek.

3. **Performance Metrics**:
   - Area Under the Receiver Operating Characteristic Curve (AUROC)
   - Precision, Recall, F1-score
   - SHAP values for feature importance

## Results
- The best-performing model achieved an AUROC of **0.81** for cognitive impairment classification.
- DGM structural connectivity and DAN participation coefficients were the most predictive features.
- External validation showed an AUROC of **0.76**, confirming model generalizability.

## Acknowledgments
This work is part of research on cognitive impairment in MS. The dataset used in this project is not publicly available due to privacy concerns.

---
For any questions or contributions, please feel free to open an issue or submit a pull request.

