
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="SteelSight AI",
    page_icon="⚙️",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM INTERFACE
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #173F5F;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 18px;
        color: #566573;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 16px;
        border-radius: 8px;
        background-color: #EAF2F8;
        border-left: 5px solid #2874A6;
        margin-bottom: 20px;
    }

    [data-testid="stMetric"] {
        background-color: #F8F9F9;
        border: 1px solid #D5DBDB;
        padding: 14px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbours": "model/knn.pkl",
    "Gaussian Naive Bayes": "model/gaussian_naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}


@st.cache_resource
def load_application_files():
    loaded_models = {
        model_name: joblib.load(model_path)
        for model_name, model_path in MODEL_FILES.items()
    }

    loaded_label_encoder = joblib.load("model/label_encoder.pkl")
    loaded_feature_columns = joblib.load("model/feature_columns.pkl")

    return loaded_models, loaded_label_encoder, loaded_feature_columns


try:
    models, label_encoder, feature_columns = load_application_files()

except Exception as error:
    st.error(f"Application files could not be loaded: {error}")
    st.stop()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<p class="main-title">SteelSight AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">'
    'Machine Learning System for Steel Plate Fault Classification'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    This application classifies steel-plate surface faults using geometric,
    luminosity and shape-related measurements. Upload the supplied test CSV
    file and select a trained machine-learning model to evaluate its performance.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Evaluation Controls")

selected_model_name = st.sidebar.selectbox(
    "Select a classification model",
    options=list(MODEL_FILES.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload test data",
    type=["csv"],
    help="Upload test_data.csv containing 27 features and the Fault_Type column."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: UCI Steel Plates Faults | "
    "Classes: 7 | Input features: 27"
)


# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------

if uploaded_file is None:
    st.info(
        "Upload `test_data.csv` using the sidebar to begin model evaluation."
    )

    st.subheader("Available Models")

    available_models = pd.DataFrame({
        "Model": list(MODEL_FILES.keys()),
        "Status": ["Ready"] * len(MODEL_FILES)
    })

    st.dataframe(
        available_models,
        width="stretch",
        hide_index=True
    )

    st.stop()


try:
    uploaded_data = pd.read_csv(uploaded_file)

except Exception as error:
    st.error(f"Unable to read the uploaded CSV file: {error}")
    st.stop()


required_columns = feature_columns + ["Fault_Type"]
missing_columns = [
    column for column in required_columns
    if column not in uploaded_data.columns
]

if missing_columns:
    st.error(
        "The uploaded file is missing the following required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


evaluation_data = uploaded_data[required_columns].copy()

if evaluation_data.isnull().any().any():
    st.error("The uploaded test data contains missing values.")
    st.stop()


unknown_classes = set(
    evaluation_data["Fault_Type"].unique()
) - set(label_encoder.classes_)

if unknown_classes:
    st.error(
        "Unknown fault classes found: "
        + ", ".join(sorted(unknown_classes))
    )
    st.stop()


X_uploaded = evaluation_data[feature_columns]
y_uploaded = label_encoder.transform(
    evaluation_data["Fault_Type"]
)

selected_model = models[selected_model_name]

predicted_classes = selected_model.predict(X_uploaded)
predicted_probabilities = selected_model.predict_proba(X_uploaded)


# ---------------------------------------------------------
# CALCULATE METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(y_uploaded, predicted_classes)

auc = roc_auc_score(
    y_uploaded,
    predicted_probabilities,
    multi_class="ovr",
    average="weighted"
)

precision = precision_score(
    y_uploaded,
    predicted_classes,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_uploaded,
    predicted_classes,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_uploaded,
    predicted_classes,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_uploaded,
    predicted_classes
)


# ---------------------------------------------------------
# DISPLAY DATASET
# ---------------------------------------------------------

st.subheader("Uploaded Test Dataset")

dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

dataset_col1.metric("Test Records", evaluation_data.shape[0])
dataset_col2.metric("Input Features", len(feature_columns))
dataset_col3.metric(
    "Fault Classes",
    evaluation_data["Fault_Type"].nunique()
)

with st.expander("Preview uploaded test data"):
    st.dataframe(
        evaluation_data.head(20),
        width="stretch"
    )


# ---------------------------------------------------------
# DISPLAY METRICS
# ---------------------------------------------------------

st.subheader(f"Evaluation Results: {selected_model_name}")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col4, metric_col5, metric_col6 = st.columns(3)

metric_col1.metric("Accuracy", f"{accuracy:.4f}")
metric_col2.metric("AUC", f"{auc:.4f}")
metric_col3.metric("Precision", f"{precision:.4f}")
metric_col4.metric("Recall", f"{recall:.4f}")
metric_col5.metric("F1 Score", f"{f1:.4f}")
metric_col6.metric("MCC", f"{mcc:.4f}")


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

st.subheader("Confusion Matrix")

class_numbers = np.arange(len(label_encoder.classes_))

cm = confusion_matrix(
    y_uploaded,
    predicted_classes,
    labels=class_numbers
)

figure, axis = plt.subplots(figsize=(9, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    ax=axis
)

axis.set_title(f"Confusion Matrix – {selected_model_name}")
axis.set_xlabel("Predicted Fault Class")
axis.set_ylabel("Actual Fault Class")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

st.pyplot(figure)


# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_uploaded,
    predicted_classes,
    labels=class_numbers,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_dataframe = pd.DataFrame(report).transpose()
st.dataframe(report_dataframe.round(4), width="stretch")


# ---------------------------------------------------------
# PREDICTION RESULTS
# ---------------------------------------------------------

st.subheader("Prediction Results")

prediction_output = evaluation_data.copy()

prediction_output["Predicted_Fault_Type"] = (
    label_encoder.inverse_transform(predicted_classes)
)

prediction_output["Prediction_Status"] = np.where(
    prediction_output["Fault_Type"]
    == prediction_output["Predicted_Fault_Type"],
    "Correct",
    "Incorrect"
)

st.dataframe(
    prediction_output[
        [
            "Fault_Type",
            "Predicted_Fault_Type",
            "Prediction_Status"
        ]
    ],
    width="stretch"
)

prediction_csv = prediction_output.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Prediction Results",
    data=prediction_csv,
    file_name="steel_fault_predictions.csv",
    mime="text/csv"
)
