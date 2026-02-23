import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Simon Bank HELOC Tool", layout="wide")
st.title("🏦 Simon Bank: HELOC Eligibility Screener")
st.markdown("### Decision Support System for Credit Risk Assessment")

@st.cache_resource
def load_model():
    with open("heloc_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_medians():
    # IMPORTANT: this file must be present in your Streamlit repo
    df = pd.read_csv("heloc_dataset_v1.csv")

    # match training: drop redundant column if you did in training
    if "NumInqLast6M" in df.columns:
        df = df.drop(columns=["NumInqLast6M"])

    # drop target
    df = df.drop(columns=["RiskPerformance"])

    # match training spirit: treat special codes as missing before medians
    df = df.replace([-7, -8, -9], np.nan)

    med = df.median(numeric_only=True)
    return med

model = load_model()
medians = load_medians()

st.sidebar.header("Applicant Profile")
st.sidebar.write("Adjust features to see prediction impact.")

ext_risk = st.sidebar.slider("External Risk Estimate", 30, 100, 75)
sat_trades = st.sidebar.number_input("Number of Satisfactory Trades", 0, 100, 20)
rev_burden = st.sidebar.slider("Net Fraction Revolving Burden (Usage %)", 0, 150, 30)
never_delq = st.sidebar.radio("Never Delinquent?", ["Yes", "No"])

if st.button("Generate Eligibility Prediction"):
    never_delq_val = 1 if never_delq == "Yes" else 0

    user_inputs = {
        "ExternalRiskEstimate": ext_risk,
        "NumSatisfactoryTrades": sat_trades,
        "NetFractionRevolvingBurden": rev_burden,
        "NeverDelinquent": never_delq_val,
    }

    # 1) Figure out what columns the model expects
    full_columns = getattr(model, "feature_names_in_", None)
    if full_columns is None:
        st.error(
            "Your saved object does not expose feature_names_in_. "
            "Quick fix: re-save the model after fitting on a DataFrame, or hardcode the column list."
        )
        st.stop()

    # 2) Start from medians (better than zeros), aligned to full_columns
    base = medians.reindex(full_columns).fillna(0.0)

    input_df = pd.DataFrame([base], columns=full_columns)

    # 3) Overwrite with user inputs
    for col, val in user_inputs.items():
        if col in input_df.columns:
            input_df.at[0, col] = float(val)
        else:
            st.warning(f"Column '{col}' not found in training features.")

    # 4) Force numeric dtype (THIS FIXES THE IMPUTER CRASH)
    input_df = input_df.astype(float)

    # Debug helpers (you can remove later)
    st.caption(f"Model classes_: {getattr(model, 'classes_', 'N/A')}")
    st.caption(f"Input dtypes unique: {set(input_df.dtypes.astype(str))}")

    probability_bad = float(model.predict_proba(input_df)[0][1])  # assumes class 1 = Bad

    st.divider()
    st.write(f"**Risk score (P(Bad))**: {probability_bad:.2%}")

    if probability_bad < 0.5:
        st.success(f"### Result: APPROVED / POSITIVE (Confidence: {1 - probability_bad:.2%})")
        st.balloons()
    else:
        st.error(f"### Result: DECLINED / NEGATIVE (Risk Score: {probability_bad:.2%})")