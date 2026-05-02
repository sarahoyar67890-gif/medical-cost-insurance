import streamlit as st
import numpy as np
import pickle

# load model
with open("insurance_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Insurance Cost Prediction")

# inputs
age = st.number_input("Age")
sex = st.selectbox("Sex (0 = male, 1 = female)", [0, 1])
bmi = st.number_input("BMI")
children = st.number_input("Children", step=1)
smoker = st.selectbox("Smoker (0 = yes, 1 = no)", [0, 1])
region = st.selectbox("Region (0=southeast,1=southwest,2=northeast,3=northwest)", [0, 1, 2, 3])

# prediction button
if st.button("Predict"):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)
    
    st.write("Predicted Insurance Cost (USD):")
    st.write(prediction[0])
    st.write(model.n_features_in_)
