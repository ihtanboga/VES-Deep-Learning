# Comparative Evaluation of Machine Learning Models for Predicting Treatment Non-response in Idiopathic Premature Ventricular Contractions

A multicenter study comparing six machine learning approaches — Logistic Regression, MLP, XGBoost, TabTransformer, TabNet, and Kolmogorov-Arnold Network (KAN) — for predicting pharmacological treatment non-response in idiopathic PVC patients.

## Key Results (External Validation, N=366)

| Model | AUROC | Brier Score | ICI | F1 Score |
|-------|-------|-------------|-----|----------|
| XGBoost | 0.987 | 0.034 | 0.018 | 0.930 |
| KAN | 0.971 | 0.068 | 0.053 | 0.872 |
| TabTransformer | 0.959 | 0.055 | 0.023 | 0.885 |
| TabNet | 0.958 | 0.064 | 0.046 | 0.868 |
| MLP | 0.946 | 0.077 | 0.025 | 0.833 |
| Logistic Regression | 0.927 | 0.090 | 0.032 | 0.782 |

## Clinical Calculator

An open-access web-based prediction tool is available at:

**https://huggingface.co/spaces/halil21/ipvc-treatment-predictor**

Clinicians can input patient-specific parameters and obtain individualized risk predictions from multiple models, with no software installation required.

## Project Structure

```
VES-Deep-Learning/
├── notebooks/
│   └── VES_deep_learning_revised.ipynb   # Training & evaluation notebook
├── r_scripts/
│   ├── analysis_revised.R                # Performance metrics & figures
│   └── perform33.R                       # 34-metric evaluation functions
├── model_weights/                        # Trained model checkpoints
│   ├── logistic_regression_model.pkl
│   ├── xgboost_model.pkl
│   ├── tabtransformer_model.pth
│   ├── kan_model.pth
│   ├── mlp_model.pkl
│   ├── tabnet_model.zip
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── figures/                              # All publication figures
│   ├── figure2_roc_pre_recalib.*         # ROC curves
│   ├── figure3_calibration_pre_recalib.* # Calibration plots
│   ├── figure4_dca_pre_recalib.*         # Decision curve analysis
│   ├── figure5_calibration_post_recalib.*
│   ├── figure6_roc_post_recalib.*
│   ├── figure7_dca_post_recalib.*
│   ├── shap_*.png                        # SHAP summary plots
│   └── supp_*.png                        # Supplementary figures
├── requirements.txt
└── .gitignore
```

## Setup

### Python Environment

```bash
git clone https://github.com/ihtanboga/VES-Deep-Learning.git
cd VES-Deep-Learning
pip install -r requirements.txt
```

### Running the Analysis

1. **Training & Evaluation (Python):**
   Open `notebooks/VES_deep_learning_revised.ipynb` in Jupyter or Google Colab.

2. **Performance Metrics & Figures (R):**
   ```bash
   Rscript r_scripts/analysis_revised.R
   ```
   Required R packages: `pROC`, `CalibrationCurves`, `ggplot2`, `dplyr`, `dcurves`, `gridExtra`, `viridis`, `boot`, `tidyr`

## Methodology

- **Training cohort:** 1,278 patients from 8 centers across Istanbul
- **External validation cohort:** 366 patients from 3 centers in the Anatolia region
- **Internal validation:** Stratified 80/20 train-validation split within Istanbul cohort for deep learning models; 5-fold cross-validation for classical ML models
- **Preprocessing:** Label encoding and standard scaling fit exclusively on training data
- **Recalibration:** Logistic recalibration (Platt scaling) on Istanbul validation subset
- **Sensitivity analysis:** Models retrained without PVC burden to assess feature dependency

## Features (20 predictors)

**Numeric (14):** PVC burden (%), PVC QRS duration, LVEF, Age, PVC prematurity index, QRS ratio, Mean heart rate, Symptom duration, QTc sinus, PVC coupling interval dispersion, CI variability, PVC peak QRS duration, PVC coupling interval, PVC compensatory interval

**Categorical (6):** Multifocal PVC, Non-sustained VT, Gender, Hypertension, Diabetes mellitus, Full compensation

## Citation

If you use this code or the clinical calculator, please cite:

> Tanboga IH, et al. Comparative Evaluation of Machine Learning Models Including TabTransformer and Kolmogorov-Arnold Networks for Predicting Treatment Non-response in Idiopathic Premature Ventricular Contractions: A Multicenter Study. *[Journal]*, 2025.

## Contact

For questions, contact [haliltanboga@yahoo.com](mailto:haliltanboga@yahoo.com).

## License

This project is provided for academic and research purposes.
