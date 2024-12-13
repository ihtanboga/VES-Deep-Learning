# Premature Ventricular Depolarization Deep Learning Project

## Description
This project evaluates various machine learning models such as Logistic Regression, TabNet, TabTransformer, MLP, and Kolmogorov-Arnold Network for a classification task.

## Project Structure
```
VES-Deep-Learning/
|
├── main.ipynb                  # Main notebook for analysis and training
├── requirements.txt            # Python dependencies
├── README.md                   # Project description and setup instructions
├── data/                       # Folder for dataset(s)
│   ├── df.xlsx                 # Example dataset (11% of the real dataset)
├── models/                     # Folder for saved models and encoders
│   ├── logistic_regression_label_encoder.pkl
│   ├── logistic_regression_model.pkl
│   ├── logistic_regression_scaler.pkl
│   ├── kan_model.pth
│   ├── mlp_model.pkl
│   ├── mlp_scaler.pkl
│   ├── tabnet_model.pth
│   ├── tabnet_scaler.joblib
│   ├── tabtransformer_model.pth
│   ├── tabtransformer_model_scaler.pkl
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd VES-Deep-Learning
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in the `data/` directory and ensure the file is named `df.xlsx`.

4. Open the Jupyter Notebook `main.ipynb` and follow the steps.

## Notes
- Ensure the saved models in the `models/` directory are used in the notebook for evaluation.
- For any questions, contact [haliltanboga@yahoo.com].

