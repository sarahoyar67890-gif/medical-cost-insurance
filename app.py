import streamlit as st
import numpy as np
import pickle
import pandas as pd

# load model
with open("insurance_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Insurance Cost Prediction 💰")

# inputs
age = st.number_input("Age")
bmi = st.number_input("BMI")
children = st.number_input("Children", step=1)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# prediction
if st.button("Predict"):

    # 🔥 recreate EXACT feature engineering from training
    input_dict = {
        "age": age,
        "bmi": bmi,
        "children": children,

        "sex_male": 1 if sex == "male" else 0,

        "smoker_yes": 1 if smoker == "yes" else 0,

        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,

        # feature engineering you added in training
        "smoker_bmi": (1 if smoker == "yes" else 0) * bmi
    }

    # convert to DataFrame (IMPORTANT ❗)
    input_data = pd.DataFrame([input_dict])

    # ensure column order matches training
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)

    st.success(f"Predicted Insurance Cost: ${prediction[0]:.2f}")
