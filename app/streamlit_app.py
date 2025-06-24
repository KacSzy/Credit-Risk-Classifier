import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Set page config
st.set_page_config(page_title="Credit Risk Classification", page_icon="💲", layout="wide")

# Constants
NUMERICAL_FEATURES = [
    'duration_months',
    'credit_amount',
    'installment_rate',
    'age',
    'present_residence_since',
    'number_existing_credits',
    'people_liable'
]

FEATURE_COLUMNS = [
    'status_account', 'duration_months', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_since', 'installment_rate', 'personal_status',
    'other_debtors', 'present_residence_since', 'property', 'age',
    'other_installment_plans', 'housing', 'number_existing_credits', 'job',
    'people_liable', 'telephone', 'foreign_worker', 'sex'
]

# Category mappings for one-hot encoding
CATEGORY_MAPS = {
    'status_account': {
        '... < 0 DM': '0__=_...___200_DM',
        '0 <= ... < 200 DM': '0__=_...___200_DM',
        '>= 200 DM / salary': '__=_200_DM_/_salary',
        'no checking account': 'no_checking_account'
    },
    'credit_history': {
        'critical account/other credits existing (not at this bank)': 'critical_account/other_credits_existing_(not_at_this_bank)',
        'delay in paying off in the past': 'delay_in_paying_off_in_the_past',
        'existing credits paid back duly till now': 'existing_credits_paid_back_duly_till_now',
        'no credits taken/all credits paid back duly': 'no_credits_taken/all_credits_paid_back_duly'
    },
    'purpose': {
        'car (new)': 'car_(new)',
        'car (used)': 'car_(used)',
        'domestic appliances': 'domestic_appliances',
        'education': 'education',
        'furniture/equipment': 'furniture/equipment',
        'others': 'others',
        'radio/television': 'radio/television',
        'repairs': 'repairs',
        'retraining': 'retraining'
    },
    'savings_account': {
        '... < 100 DM': '...__=_100_DM',
        '100 <= ... < 500 DM': '100__=_...___500_DM',
        '500 <= ... < 1000 DM': '500__=_...___1000_DM',
        'unknown / no savings account': 'unknown/no_savings_account'
    },
    'employment_since': {
        '... < 1 year': '...__=_1_year',
        '1 <= ... < 4 years': '1__=_...___4_years',
        '4 <= ... < 7 years': '4__=_...___7_years',
        '>= 7 years': '...__=_7_years',
        'unemployed': 'unemployed'
    },
    'personal_status': {
        'divorced/separated/married': 'divorced/separated/married',
        'married/widowed': 'married/widowed',
        'single': 'single'
    },
    'other_debtors': {
        'guarantor': 'guarantor',
        'none': 'none'
    },
    'property': {
        'car or other, not in attribute 6': 'car_or_other__not_in_attribute_6',
        'real estate': 'real_estate',
        'unknown / no property': 'unknown/no_property'
    },
    'other_installment_plans': {
        'bank': 'bank',
        'none': 'none',
        'stores': 'stores'
    },
    'housing': {
        'for free': 'for_free',
        'own': 'own',
        'rent': 'rent'
    },
    'job': {
        'unemployed/ unskilled - non-resident': 'unemployed/unskilled_-_non-resident',
        'unskilled - resident': 'unskilled_-_resident',
        'skilled employee/ official': 'skilled_employee/official',
        'management/ self-employed/highly qualified employee/ officer': 'management/self-employed/highly_qualified_employee/officer'
    },
    'telephone': {
        'no': 'no',
        'yes, registered under the customers name': 'yes_registered_under_the_customers_name'
    },
    'foreign_worker': {
        'no': 'no',
        'yes': 'yes'
    },
    'sex': {
        'M': 'M',
        'F': 'F'
    }
}


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the dataset."""
    data_path = Path(__file__).parents[1] / "data" / "german_decoded.csv"
    return pd.read_csv(data_path)


@st.cache_resource
def load_models() -> Dict[str, Any]:
    """Load and cache the ML models."""
    models_dir = Path(__file__).parents[1] / "models"
    rf_model = joblib.load(models_dir / "random_forest.joblib")
    xgb_model = joblib.load(models_dir / "xgboost.joblib")
    return {"Random Forest": rf_model, "XGBoost": xgb_model}


def get_feature_name(col: str, val: str) -> Optional[str]:
    """Convert categorical feature and value to model feature name."""
    if col in CATEGORY_MAPS:
        mapped_val = CATEGORY_MAPS[col].get(val)
        if mapped_val is not None:
            return f"{col}_{mapped_val}"
    return None


def prepare_input_for_model(user_input_df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Transform user input dataframe to format expected by the model."""
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    else:
        expected_features = model.get_booster().feature_names

    model_input_df = pd.DataFrame(columns=expected_features)
    model_input_df.loc[0] = 0  # zero initialization

    # Map numerical columns directly
    for col in NUMERICAL_FEATURES:
        if col in user_input_df.columns and col in expected_features:
            model_input_df.at[0, col] = user_input_df.at[0, col]

    # Map categorical with one-hot encoding
    for col in user_input_df.columns:
        if col not in NUMERICAL_FEATURES:
            val = user_input_df.at[0, col]
            feat_name = get_feature_name(col, val)
            if feat_name and feat_name in expected_features:
                model_input_df.at[0, feat_name] = 1

    return model_input_df.loc[[0]]


def display_categorical_inputs(df: pd.DataFrame, cat_columns: List[str]) -> Dict[str, Any]:
    """Display categorical input fields in the Streamlit interface."""
    selected_features = {}

    col1, col2 = st.columns(2)
    mid_point = len(cat_columns) // 2

    with col1:
        st.subheader("Categorical Features")
        for col in cat_columns[:mid_point]:
            options = sorted(df[col].dropna().unique())
            selected_features[col] = st.selectbox(
                f"{col.replace('_', ' ').title()}",
                options
            )

    with col2:
        st.subheader("Categorical Features (cont.)")
        for col in cat_columns[mid_point:]:
            options = sorted(df[col].dropna().unique())
            selected_features[col] = st.selectbox(
                f"{col.replace('_', ' ').title()}",
                options
            )

    return selected_features


def display_numerical_inputs() -> Dict[str, Any]:
    """Display numerical input fields in the Streamlit interface."""
    numerical_inputs = {}

    st.subheader("Numerical Features")
    num_col1, num_col2 = st.columns(2)

    with num_col1:
        numerical_inputs['duration_months'] = st.slider(
            "Duration in months", 1, 72, 12
        )
        numerical_inputs['credit_amount'] = st.slider(
            "Credit amount", 250, 50_000, 5_000
        )
        numerical_inputs['installment_rate'] = st.slider(
            "Installment rate (%)", 1, 5, 2
        )
        numerical_inputs['present_residence_since'] = st.slider(
            "Present residence since (years)", 0, 10, 3
        )

    with num_col2:
        numerical_inputs['age'] = st.slider(
            "Age (years)", 18, 100, 30
        )
        numerical_inputs['number_existing_credits'] = st.slider(
            "Number of existing credits", 0, 10, 1
        )
        numerical_inputs['people_liable'] = st.slider(
            "Number of people liable", 1, 5, 1
        )

    return numerical_inputs


def display_prediction_result(prediction: int, prediction_proba: List[float]) -> None:
    """Display prediction results in the Streamlit interface."""
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Good Credit Risk (Low Risk)")
        risk_score = round(prediction_proba[1] * 100)  # Probability of class 1
    else:  # 0 is bad risk
        st.error("Bad Credit Risk (High Risk)")
        risk_score = round((1 - prediction_proba[1]) * 100)

    st.metric("Risk Score", f"{risk_score}%")


def main() -> None:
    """Main application function."""
    st.title("Credit Risk Classification")
    st.markdown("This app predicts credit risk based on applicant information.")

    # Load data and models
    df = load_data()
    models = load_models()

    # Get categorical columns
    categorical_columns = [col for col in FEATURE_COLUMNS if col not in NUMERICAL_FEATURES]

    # Display input fields
    categorical_inputs = display_categorical_inputs(df, categorical_columns)
    numerical_inputs = display_numerical_inputs()

    # Combine all inputs
    all_inputs = {**categorical_inputs, **numerical_inputs}

    # Model selection
    model_choice = st.selectbox("Select Model", list(models.keys()))

    # Prediction button
    if st.button("Predict Credit Risk"):
        input_df = pd.DataFrame([all_inputs])
        model = models[model_choice]

        # Prepare input for model
        model_input_df = prepare_input_for_model(input_df, model)

        # Make prediction
        prediction = model.predict(model_input_df)[0]
        prediction_proba = model.predict_proba(model_input_df)[0]

        # Display results
        display_prediction_result(prediction, prediction_proba)


if __name__ == "__main__":
    main()
