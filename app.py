import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)


model = load_model()
metrics = load_metrics()
R = metrics["ranges"]

st.title("🩺 Diabetes Risk Predictor")
st.caption("Educational demo trained on the Pima Indians Diabetes data (UCI / Kaggle). "
           "Not a medical device and not for diagnosis.")

# ------------------------------------------------------------------ sidebar inputs
st.sidebar.header("Patient measurements")
st.sidebar.write("Move the sliders, then read the result on the **Prediction** tab.")
inputs = {
    "Pregnancies": st.sidebar.slider("Pregnancies", 0, 17, 2),
    "Glucose": st.sidebar.slider("Glucose (mg/dL, 2-hour OGTT)", 40, 200, 120),
    "BloodPressure": st.sidebar.slider("Diastolic blood pressure (mm Hg)", 30, 125, 70),
    "SkinThickness": st.sidebar.slider("Skin-fold thickness (mm)", 5, 100, 25),
    "Insulin": st.sidebar.slider("2-hour serum insulin (mu U/ml)", 10, 850, 100),
    "BMI": st.sidebar.slider("BMI", 15.0, 70.0, 28.0, 0.1),
    "DiabetesPedigreeFunction": st.sidebar.slider("Diabetes pedigree function", 0.05, 2.5, 0.4, 0.01),
    "Age": st.sidebar.slider("Age (years)", 18, 85, 33),
}
row = pd.DataFrame([inputs])
prob = float(model.predict_proba(row)[0, 1])

tab1, tab2, tab3 = st.tabs(["Prediction", "Model performance", "About the data"])

# ------------------------------------------------------------------ tab 1: prediction
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Estimated risk")
        st.metric("Probability of diabetes", f"{prob:.0%}")
        st.progress(prob)
        if prob < 0.30:
            st.success("Lower risk according to the model.")
        elif prob < 0.60:
            st.warning("Moderate risk according to the model.")
        else:
            st.error("Higher risk according to the model.")
        st.caption("A screening estimate from 768 women. Please see a doctor for any health decision.")
    with c2:
        st.subheader("What would change the result?")
        st.write("Risk when one measurement is varied and everything else is kept as set in the sidebar:")
        feature = st.selectbox("Measurement to vary", list(inputs), index=1)
        grid = np.linspace(R[feature]["min"], R[feature]["max"], 40)
        sweep = pd.DataFrame([{**inputs, feature: v} for v in grid])
        curve = pd.DataFrame({feature: grid, "risk": model.predict_proba(sweep)[:, 1]}).set_index(feature)
        st.line_chart(curve)
    with st.expander("Show the input row sent to the model"):
        st.dataframe(row)

# ------------------------------------------------------------------ tab 2: performance
with tab2:
    st.subheader(f"Chosen model: {metrics['model']}")
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC (test)", f"{metrics['test_roc_auc']:.3f}")
    b.metric("Recall (test)", f"{metrics['test_recall']:.2f}")
    c.metric("F1 (test)", f"{metrics['test_f1']:.2f}")
    d.metric("Accuracy (test)", f"{metrics['test_accuracy']:.2f}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Cross-validated ROC-AUC of the candidates**")
        st.bar_chart(pd.Series(metrics["cv_roc_auc"], name="ROC-AUC"))
        st.markdown("**Confusion matrix (test set)**")
        cm = metrics["confusion"]
        st.table(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                              index=["Actually not diabetic", "Actually diabetic"],
                              columns=["Predicted not diabetic", "Predicted diabetic"]))
    with right:
        st.markdown("**Which measurements matter most** (permutation importance)")
        st.bar_chart(pd.Series(metrics["importance"], name="importance").sort_values())
    st.info(f"Trained on {metrics['n_train']} patients and tested on {metrics['n_test']} unseen patients. "
            "With so little data, expect a wide margin of error.")

# ------------------------------------------------------------------ tab 3: about
with tab3:
    st.subheader("About the data")
    st.markdown(
        "- **Source:** Pima Indians Diabetes data from the UCI repository / Kaggle (768 women, 35% diabetic).\n"
        "- **Cleaning:** zeros in glucose, blood pressure, skin thickness, insulin and BMI are impossible, so they are treated as missing and replaced by the median.\n"
        "- **Models compared:** logistic regression, random forest and gradient boosting, chosen by 5-fold cross-validated ROC-AUC.\n"
        "- **Limits:** one small, single-population dataset. It cannot be generalised to other groups.")
    stats = pd.DataFrame(R).T.rename(columns=str.title)
    st.markdown("**Range of each measurement in the training data**")
    st.dataframe(stats)
