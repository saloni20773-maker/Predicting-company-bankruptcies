import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load Saved Model and Selected Features
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tuned_random_forest_candidate.joblib"
FEATURES_PATH = BASE_DIR / "selected_features.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_features():
    return joblib.load(FEATURES_PATH)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📊 Company Bankruptcy Prediction")
st.markdown(
    """
    ### Machine Learning–Based Financial Risk Assessment

    Upload a CSV file containing the required financial indicators
    to predict whether a company is likely to be classified as
    **Bankrupt** or **Non-Bankrupt**.
    """
)

st.divider()


# --------------------------------------------------
# Check Model Files
# --------------------------------------------------
if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error(
        "Model files are missing. Please make sure both "
        "`tuned_random_forest_candidate.joblib` and "
        "`selected_features.joblib` are present in the repository."
    )
    st.stop()


try:
    model = load_model()
    selected_features = list(load_features())
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# --------------------------------------------------
# Model Information
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Tuned Random Forest")

with col2:
    st.metric("Required Features", len(selected_features))

with col3:
    st.metric("Prediction Type", "Binary Classification")


st.divider()


# --------------------------------------------------
# Instructions
# --------------------------------------------------
st.subheader("📁 Upload Company Financial Data")

st.info(
    "Your CSV should contain the financial indicator columns used by "
    "the trained model. The app will automatically check whether all "
    "required features are available."
)


# --------------------------------------------------
# Download Template
# --------------------------------------------------
template_df = pd.DataFrame(columns=selected_features)

st.download_button(
    label="⬇️ Download CSV Template",
    data=template_df.to_csv(index=False),
    file_name="bankruptcy_prediction_template.csv",
    mime="text/csv"
)


# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        input_df = pd.read_csv(uploaded_file)

        st.subheader("📋 Uploaded Data")
        st.dataframe(input_df, use_container_width=True)

        # ------------------------------------------
        # Remove target column if included
        # ------------------------------------------
        target_column = "Bankrupt?"

        if target_column in input_df.columns:
            input_df = input_df.drop(columns=[target_column])

        # ------------------------------------------
        # Check missing features
        # ------------------------------------------
        missing_features = [
            feature
            for feature in selected_features
            if feature not in input_df.columns
        ]

        if missing_features:

            st.error(
                f"The uploaded CSV is missing {len(missing_features)} "
                "required feature(s)."
            )

            with st.expander("Show missing features"):
                for feature in missing_features:
                    st.write(f"- {feature}")

            st.stop()

        # ------------------------------------------
        # Select features in correct order
        # ------------------------------------------
        prediction_df = input_df[selected_features].copy()

        # ------------------------------------------
        # Check numeric values
        # ------------------------------------------
        non_numeric_columns = prediction_df.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        if non_numeric_columns:
            st.error(
                "The following required columns contain non-numeric "
                "values:"
            )

            for column in non_numeric_columns:
                st.write(f"- {column}")

            st.stop()

        # ------------------------------------------
        # Handle missing values
        # ------------------------------------------
        if prediction_df.isnull().any().any():

            st.error(
                "The uploaded data contains missing values. "
                "Please provide values for all required financial indicators."
            )
            st.stop()

        # ------------------------------------------
        # Prediction
        # ------------------------------------------
        predictions = model.predict(prediction_df)

        # Probability if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(prediction_df)[:, 1]
        else:
            probabilities = [None] * len(predictions)

        # ------------------------------------------
        # Create Results DataFrame
        # ------------------------------------------
        results_df = input_df.copy()

        results_df["Prediction"] = predictions

        if probabilities[0] is not None:
            results_df["Bankruptcy Probability"] = probabilities

        results_df["Prediction Label"] = results_df["Prediction"].map(
            {
                0: "Non-Bankrupt",
                1: "Bankrupt"
            }
        )

        # ------------------------------------------
        # Results
        # ------------------------------------------
        st.divider()
        st.subheader("🔍 Prediction Results")

        display_columns = [
            "Prediction Label",
            "Prediction"
        ]

        if "Bankruptcy Probability" in results_df.columns:
            display_columns.append("Bankruptcy Probability")

        st.dataframe(
            results_df[display_columns],
            use_container_width=True
        )

        # ------------------------------------------
        # Summary
        # ------------------------------------------
        bankrupt_count = int((predictions == 1).sum())
        non_bankrupt_count = int((predictions == 0).sum())

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Bankrupt",
                bankrupt_count
            )

        with col2:
            st.metric(
                "Predicted Non-Bankrupt",
                non_bankrupt_count
            )

        # ------------------------------------------
        # Detailed Result for Single Company
        # ------------------------------------------
        if len(predictions) == 1:

            st.divider()
            st.subheader("🏢 Company Risk Assessment")

            if predictions[0] == 1:

                st.error(
                    "⚠️ Prediction: Bankrupt"
                )

                if probabilities[0] is not None:
                    st.write(
                        f"Estimated bankruptcy probability: "
                        f"**{probabilities[0]:.2%}**"
                    )

            else:

                st.success(
                    "✅ Prediction: Non-Bankrupt"
                )

                if probabilities[0] is not None:
                    st.write(
                        f"Estimated bankruptcy probability: "
                        f"**{probabilities[0]:.2%}**"
                    )

        # ------------------------------------------
        # Download Results
        # ------------------------------------------
        st.divider()

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=results_df.to_csv(index=False),
            file_name="bankruptcy_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"An error occurred while processing the uploaded file: {e}"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Predicting Company Bankruptcies | Machine Learning & Data Science Project"
)
