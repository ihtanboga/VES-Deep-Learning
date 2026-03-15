"""
iPVC Treatment Non-response Prediction — Clinical Calculator
=============================================================
Gradio web app supporting 4 models:
  Logistic Regression, XGBoost, TabTransformer, KAN

Model weights and scaler.pkl are expected in the model_weights/ subdirectory.
"""

import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import gradio as gr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(APP_DIR, "model_weights")

# ---------------------------------------------------------------------------
# Feature definitions (must match notebook order exactly)
# ---------------------------------------------------------------------------
numeric_features = [
    "PVCyüzdesi",
    "PVCQRS",
    "LVEF",
    "Yaş",
    "PVCPrematurındex",
    "QRSratio",
    "OrtalamaHR",
    "SemptomSüresi",
    "QTCsinus",
    "PVCCouplingIntervaldispersiyon",
    "CIvariability",
    "PVCPeakQRSduration",
    "PVCCouplingInterval",
    "PVCCompansatuarInterval",
]

categorical_features = [
    "MultifokalPVC",
    "Non_susteinedVT",
    "Cins",
    "HT",
    "DM",
    "Fullcompansasion",
]

all_features = numeric_features + categorical_features  # total = 20

# Slider label  ->  internal feature name  (same order as numeric_features)
SLIDER_LABELS = [
    "PVC Burden (%)",
    "PVC QRS Duration (ms)",
    "LVEF (%)",
    "Age (years)",
    "PVC Prematurity Index",
    "QRS Ratio",
    "Mean Heart Rate (bpm)",
    "Symptom Duration (months)",
    "QTc Sinus (ms)",
    "PVC CI Dispersion (ms)",
    "CI Variability",
    "PVC Peak QRS Duration (ms)",
    "PVC Coupling Interval (ms)",
    "PVC Compensatory Interval (ms)",
]

RADIO_LABELS = [
    "Multifocal PVC",
    "Non-sustained VT",
    "Gender",
    "Hypertension",
    "Diabetes Mellitus",
    "Full Compensation",
]

# ---------------------------------------------------------------------------
# PyTorch model architectures (identical to notebook)
# ---------------------------------------------------------------------------

# ---- TabTransformer ----
class TabTransformer(nn.Module):
    def __init__(self, input_dim=20, num_classes=2, d_model=64, nhead=4,
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.fc(x)


# ---- KAN (Kolmogorov-Arnold Network) ----
class KolmogorovArnoldLayer(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim):
        super().__init__()
        self.inner_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, inner_dim), nn.ReLU(), nn.Linear(inner_dim, 1)
            )
            for _ in range(input_dim)
        ])
        self.outer_function = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, output_dim),
        )

    def forward(self, x):
        inner_outputs = [f(x[:, i:i + 1]) for i, f in enumerate(self.inner_functions)]
        return self.outer_function(torch.cat(inner_outputs, dim=1))


class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, input_dim=20, hidden_dims=None, inner_dim=37, dropout=0.467):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [94, 55]
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(KolmogorovArnoldLayer(prev_dim, inner_dim, hd))
            prev_dim = hd
        self.kan_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dims[-1], 2)

    def forward(self, x):
        for layer in self.kan_layers:
            x = self.dropout(layer(x))
        return self.output_layer(x)


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

def _load_scaler():
    path = os.path.join(WEIGHTS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"scaler.pkl not found in {WEIGHTS_DIR}. "
            "Copy scaler.pkl from the training outputs into model_weights/."
        )
    return joblib.load(path)


def _load_sklearn_model(filename):
    path = os.path.join(WEIGHTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found in {WEIGHTS_DIR}.")
    return joblib.load(path)


def _load_tabtransformer():
    path = os.path.join(WEIGHTS_DIR, "tabtransformer_model.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"tabtransformer_model.pth not found in {WEIGHTS_DIR}.")
    model = TabTransformer(
        input_dim=20, num_classes=2, d_model=64, nhead=4,
        num_layers=3, dropout=0.1
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_kan():
    path = os.path.join(WEIGHTS_DIR, "kan_model.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"kan_model.pth not found in {WEIGHTS_DIR}.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = KolmogorovArnoldNetwork(
        input_dim=20, hidden_dims=[94, 55], inner_dim=37, dropout=0.467
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Lazy-loaded cache so the models are only read once
_cache = {}


def _get(key, loader, *args):
    if key not in _cache:
        _cache[key] = loader(*args)
    return _cache[key]


# ---------------------------------------------------------------------------
# Categorical encoding helper
# ---------------------------------------------------------------------------

def _encode_categorical(value: str) -> int:
    """Encode radio-button value to integer.

    Mapping (matches LabelEncoder fit on training data):
      'No'  -> 0,  'Yes'    -> 1
      'Female' -> 0, 'Male' -> 1
    """
    mapping = {"No": 0, "Yes": 1, "Female": 0, "Male": 1}
    return mapping[value]


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def predict(
    model_choice,
    pvc_burden, pvc_qrs, lvef, age, pvc_prematur_index,
    qrs_ratio, mean_hr, symptom_duration, qtc_sinus,
    pvc_ci_dispersion, ci_variability, pvc_peak_qrs,
    pvc_coupling_interval, pvc_compensatory_interval,
    multifocal_pvc, nonsustained_vt, gender,
    hypertension, diabetes, full_compensation,
):
    try:
        scaler = _get("scaler", _load_scaler)

        # -- Build numeric array (14 features) in the correct order --
        numeric_values = np.array([[
            pvc_burden,
            pvc_qrs,
            lvef,
            age,
            pvc_prematur_index,
            qrs_ratio,
            mean_hr,
            symptom_duration,
            qtc_sinus,
            pvc_ci_dispersion,
            ci_variability,
            pvc_peak_qrs,
            pvc_coupling_interval,
            pvc_compensatory_interval,
        ]], dtype=np.float64)

        # Scale numeric features using the training scaler
        numeric_scaled = scaler.transform(numeric_values)

        # -- Build categorical array (6 features) --
        cat_values = np.array([[
            _encode_categorical(multifocal_pvc),
            _encode_categorical(nonsustained_vt),
            _encode_categorical(gender),
            _encode_categorical(hypertension),
            _encode_categorical(diabetes),
            _encode_categorical(full_compensation),
        ]], dtype=np.float64)

        # Concatenate: numeric (scaled) + categorical  -> (1, 20)
        x = np.hstack([numeric_scaled, cat_values])

        # -- Predict probability --
        if model_choice == "Logistic Regression":
            model = _get("lr", _load_sklearn_model, "logistic_regression_model.pkl")
            prob = float(model.predict_proba(x)[0, 1])

        elif model_choice == "XGBoost":
            model = _get("xgb", _load_sklearn_model, "xgboost_model.pkl")
            prob = float(model.predict_proba(x)[0, 1])

        elif model_choice == "TabTransformer":
            model = _get("tt", _load_tabtransformer)
            with torch.no_grad():
                tensor_x = torch.FloatTensor(x)
                logits = model(tensor_x)
                prob = float(torch.softmax(logits, dim=1)[0, 1].item())

        elif model_choice == "KAN":
            model = _get("kan", _load_kan)
            with torch.no_grad():
                tensor_x = torch.FloatTensor(x)
                logits = model(tensor_x)
                prob = float(torch.softmax(logits, dim=1)[0, 1].item())

        else:
            return "Error: Unknown model selected.", "", ""

        # -- Risk stratification --
        pct = prob * 100.0
        if pct < 20.0:
            risk = "LOW RISK"
        elif pct <= 40.0:
            risk = "MODERATE RISK"
        else:
            risk = "HIGH RISK"

        # -- Interpretation --
        interpretation = _build_interpretation(model_choice, pct, risk)

        probability_text = f"{pct:.1f}%"
        risk_text = f"{risk} (< 20% Low | 20-40% Moderate | > 40% High)"

        return probability_text, risk_text, interpretation

    except FileNotFoundError as e:
        return str(e), "", ""
    except Exception as e:
        return f"Prediction error: {e}", "", ""


def _build_interpretation(model_name: str, pct: float, risk: str) -> str:
    """Return a short clinical interpretation paragraph."""
    lines = [
        f"Using the {model_name} model, the predicted probability of "
        f"treatment non-response (iPVC persistence) is {pct:.1f}%.",
    ]
    if risk == "LOW RISK":
        lines.append(
            "This patient falls in the LOW risk category (< 20%). "
            "The model suggests a favorable response to anti-arrhythmic "
            "or ablation therapy is likely. Standard follow-up is recommended."
        )
    elif risk == "MODERATE RISK":
        lines.append(
            "This patient falls in the MODERATE risk category (20-40%). "
            "There is an intermediate likelihood of treatment non-response. "
            "Close monitoring and potential therapy optimization should be considered."
        )
    else:
        lines.append(
            "This patient falls in the HIGH risk category (> 40%). "
            "The model indicates a substantial probability of treatment "
            "non-response. Intensified management strategies, combination "
            "therapy, or early referral for catheter ablation may be warranted."
        )
    lines.append(
        "Note: This calculator is intended for research and clinical "
        "decision support only. It should not replace clinical judgment."
    )
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(
        title="iPVC Non-response Predictor",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# iPVC Treatment Non-response Prediction Calculator\n"
            "Enter patient parameters below and select a prediction model. "
            "The tool estimates the probability that the patient will **not respond** "
            "to iPVC treatment (anti-arrhythmic / ablation therapy)."
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=[
                    "Logistic Regression",
                    "XGBoost",
                    "TabTransformer",
                    "KAN",
                ],
                value="Logistic Regression",
                label="Prediction Model",
            )

        gr.Markdown("## Numeric Parameters")

        with gr.Row():
            pvc_burden = gr.Slider(
                minimum=0, maximum=100, step=0.1, value=15.0,
                label="PVC Burden (%)",
            )
            pvc_qrs = gr.Slider(
                minimum=80, maximum=300, step=1, value=140,
                label="PVC QRS Duration (ms)",
            )
            lvef = gr.Slider(
                minimum=10, maximum=80, step=1, value=55,
                label="LVEF (%)",
            )
        with gr.Row():
            age = gr.Slider(
                minimum=18, maximum=100, step=1, value=50,
                label="Age (years)",
            )
            pvc_prematur_index = gr.Slider(
                minimum=0.0, maximum=2.0, step=0.01, value=0.75,
                label="PVC Prematurity Index",
            )
            qrs_ratio = gr.Slider(
                minimum=0.5, maximum=3.0, step=0.01, value=1.2,
                label="QRS Ratio",
            )
        with gr.Row():
            mean_hr = gr.Slider(
                minimum=40, maximum=200, step=1, value=75,
                label="Mean Heart Rate (bpm)",
            )
            symptom_duration = gr.Slider(
                minimum=0, maximum=360, step=1, value=12,
                label="Symptom Duration (months)",
            )
            qtc_sinus = gr.Slider(
                minimum=300, maximum=600, step=1, value=420,
                label="QTc Sinus (ms)",
            )
        with gr.Row():
            pvc_ci_dispersion = gr.Slider(
                minimum=0, maximum=300, step=1, value=50,
                label="PVC CI Dispersion (ms)",
            )
            ci_variability = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, value=0.10,
                label="CI Variability",
            )
            pvc_peak_qrs = gr.Slider(
                minimum=80, maximum=300, step=1, value=140,
                label="PVC Peak QRS Duration (ms)",
            )
        with gr.Row():
            pvc_coupling_interval = gr.Slider(
                minimum=200, maximum=800, step=1, value=450,
                label="PVC Coupling Interval (ms)",
            )
            pvc_compensatory_interval = gr.Slider(
                minimum=400, maximum=1500, step=1, value=900,
                label="PVC Compensatory Interval (ms)",
            )

        gr.Markdown("## Categorical Parameters")

        with gr.Row():
            multifocal_pvc = gr.Radio(
                choices=["No", "Yes"], value="No", label="Multifocal PVC"
            )
            nonsustained_vt = gr.Radio(
                choices=["No", "Yes"], value="No", label="Non-sustained VT"
            )
            gender = gr.Radio(
                choices=["Female", "Male"], value="Male", label="Gender"
            )
        with gr.Row():
            hypertension = gr.Radio(
                choices=["No", "Yes"], value="No", label="Hypertension"
            )
            diabetes = gr.Radio(
                choices=["No", "Yes"], value="No", label="Diabetes Mellitus"
            )
            full_compensation = gr.Radio(
                choices=["No", "Yes"], value="No", label="Full Compensation"
            )

        gr.Markdown("## Prediction Results")

        with gr.Row():
            out_prob = gr.Textbox(label="Predicted Probability", interactive=False)
            out_risk = gr.Textbox(label="Risk Category", interactive=False)
        out_interp = gr.Textbox(
            label="Clinical Interpretation", interactive=False, lines=5
        )

        predict_btn = gr.Button("Predict", variant="primary")

        predict_btn.click(
            fn=predict,
            inputs=[
                model_dropdown,
                pvc_burden, pvc_qrs, lvef, age, pvc_prematur_index,
                qrs_ratio, mean_hr, symptom_duration, qtc_sinus,
                pvc_ci_dispersion, ci_variability, pvc_peak_qrs,
                pvc_coupling_interval, pvc_compensatory_interval,
                multifocal_pvc, nonsustained_vt, gender,
                hypertension, diabetes, full_compensation,
            ],
            outputs=[out_prob, out_risk, out_interp],
        )

        gr.Markdown(
            "---\n"
            "*This tool is for research and clinical decision support purposes only. "
            "Predictions should be interpreted in the context of the full clinical picture.*"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
